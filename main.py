# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import sys
import atexit
import time
from pathlib import Path

from utils.config_paths import get_logger
from utils.system import (
    APP_VERSION,
    SPLASH_DURATION_MS,
    acquire_single_instance_lock,
    release_single_instance_lock,
    shutdown_all,
    notify,
    load_settings,
    settings, settings_lock,
    start_discovery_listener,
    parse_cli_args, transcribe_cli,
    apply_backend_cli_config,
    CLIENT_ONLY_BUILD,
    start_server,
    run_tray,
    apply_auto_setup,
)

from utils.gui import (
    show_splash_screen,
    ensure_mode_selected,
    ensure_management_ui_thread,
    show_startup_error,
)
from utils.models import (
    initialize_transcriber,
    ensure_model_ready_for_local_server,
    ensure_initial_model_installation,
)

logger = get_logger(__name__)


def _report_invalid_backend_configuration(exc: Exception) -> None:
    message = f"Invalid backend configuration: {exc}"
    logger.error(message)
    try:
        if sys.stderr is not None:
            print(message, file=sys.stderr)
    except Exception:
        logger.debug("Could not write startup configuration error to stderr", exc_info=True)
    try:
        show_startup_error("Invalid backend configuration", str(exc))
    except Exception:
        logger.exception("Could not display startup configuration error")


def main(argv: list[str]) -> int:
    logger.info("CtrlSpeak starting up (version %s)", APP_VERSION)
    args = parse_cli_args(argv)
    logger.debug("Parsed CLI arguments: %s", args)

    if args.show_version:
        print(APP_VERSION)
        return 0

    if args.health_check_file:
        target = Path(args.health_check_file).expanduser().resolve()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    {
                        "product": "ctrlspeak",
                        "version": APP_VERSION,
                        "executable": str(Path(sys.executable).resolve()),
                        "frozen": bool(getattr(sys, "frozen", False)),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return 0
        except OSError:
            logger.exception("Packaged health-check result could not be written to %s", target)
            return 6

    if args.apply_update:
        from utils.update_helper import apply_update_transaction
        from utils.update_manager import UpdateError

        try:
            return apply_update_transaction(args.apply_update)
        except UpdateError as exc:
            logger.error("External updater failed [%s]: %s", exc.code, exc.user_message)
            return 7
        except Exception:
            logger.exception("External updater failed unexpectedly")
            return 7

    if args.uninstall:
        logger.info("Uninstall flag detected; launching uninstall workflow")
        from utils.system import initiate_self_uninstall
        initiate_self_uninstall(None)
        return 0

    if args.force_sendinput:
        logger.info("Force SendInput mode enabled via CLI")
        from utils.system import set_force_sendinput
        set_force_sendinput(True)

    if args.automation_flow:
        logger.info("Starting automation flow from CLI request")
        from utils.automation import run_automation_flow
        return run_automation_flow()

    # Single instance
    if not acquire_single_instance_lock():
        logger.warning("Another CtrlSpeak instance appears to be running; exiting")
        notify("CtrlSpeak is already running.")
        return 0

    # Load settings early
    logger.info("Loading configuration settings")
    load_settings()

    from utils.transcription_backend import (
        BackendPersistenceError,
        activate_runtime_backend_config,
        get_backend_config,
        initialize_openai_api_key_from_secure_storage,
        uses_bundled_runtime,
    )
    try:
        if apply_backend_cli_config(args):
            return 0
        backend_config = get_backend_config()
        activate_runtime_backend_config(backend_config)
    except (ValueError, BackendPersistenceError) as exc:
        _report_invalid_backend_configuration(exc)
        return 2

    try:
        key_loaded = initialize_openai_api_key_from_secure_storage()
        logger.info(
            "Secure OpenAI credential status: %s",
            "loaded" if key_loaded else "not configured",
        )
    except Exception:
        # A native credential-store problem must not stop private/local routes.
        logger.exception("Unable to load the secure OpenAI credential")

    bundled_runtime = uses_bundled_runtime(backend_config)
    logger.info("Selected transcription backend: %s", backend_config.backend)

    if getattr(args, "cuda_only", False):
        from utils.cuda_probe import automatic_runtime_install_supported
        from utils.models import (
            ensure_cuda_runtime_from_existing,
            install_cuda_runtime_with_progress,
            cuda_driver_available,
            cuda_runtime_ready,
        )

        if not cuda_driver_available():
            logger.error("CUDA setup requested but no CUDA-capable GPU was detected on this system.")
            try:
                print("CUDA setup aborted: no CUDA-capable GPU detected.", file=sys.stderr)
            except Exception:
                logger.debug("Failed to write CUDA hardware warning to stderr", exc_info=True)
            return 1

        if not automatic_runtime_install_supported():
            success = cuda_runtime_ready(ignore_preference=True, quiet=True)
            message = (
                "Linux system CUDA is ready for CtrlSpeak."
                if success
                else "Linux system CUDA is not ready. CtrlSpeak does not install system GPU "
                     "drivers or CUDA/cuDNN libraries; see packaging/BUILDING.md."
            )
            try:
                print(message, file=sys.stdout if success else sys.stderr)
            except Exception:
                logger.debug("Failed to write Linux CUDA readiness result", exc_info=True)
            return 0 if success else 1

        success = ensure_cuda_runtime_from_existing()
        if not success:
            success = install_cuda_runtime_with_progress(parent=None) and ensure_cuda_runtime_from_existing()
        return 0 if success else 1

    if args.auto_setup:
        logger.info("Applying auto-setup profile: %s", args.auto_setup)
        apply_auto_setup(args.auto_setup)

    # CLI: offline transcription of a file
    if args.transcribe:
        logger.info("Running CLI transcription for %s", args.transcribe)
        return transcribe_cli(args.transcribe)

    # Splash
    logger.debug("Displaying splash screen for %sms", SPLASH_DURATION_MS)
    show_splash_screen(SPLASH_DURATION_MS)

    # Prepare the shared management UI root before any background tasks need it
    logger.debug("Ensuring management UI thread is initialized")
    ensure_management_ui_thread()

    # Legacy client/server mode applies only to the bundled backend.
    if bundled_runtime:
        logger.debug("Ensuring operating mode is selected")
        ensure_mode_selected()

    # API mode is deliberately independent of bundled model assets.
    if bundled_runtime:
        if not ensure_initial_model_installation():
            logger.error("Initial model installation failed or was aborted")
            return 0

    # Determine the selected mode now that setup is complete
    with settings_lock:
        mode = settings.get("mode")
    logger.info("CtrlSpeak running in '%s' mode", mode)

    # Automatically prepare local transcription assets when running the server locally
    if bundled_runtime and mode == "client_server":
        logger.info("Preparing local transcription assets for server mode")
        if not ensure_model_ready_for_local_server():
            logger.error("Failed to prepare local model for server mode")
            return 0

    # Start discovery listener for client mode visibility
    if bundled_runtime:
        logger.debug("Starting discovery listener")
        start_discovery_listener()

    if bundled_runtime and mode == "client_server":
        logger.info("Initializing transcriber in background for warm-up")
        initialize_transcriber(interactive=False)   # warm-up local model when assets are ready
        logger.info("Starting local transcription server")
        start_server()
    elif bundled_runtime and mode == "client":
        logger.debug("Client mode selected; allowing discovery broadcast to populate")
        time.sleep(1.0)  # small delay so discovery has time to populate

    logger.info("Launching system tray UI")
    if args.post_update:
        from utils.update_helper import load_release_metadata, write_post_update_health
        from utils.update_manager import UpdateError

        try:
            write_post_update_health(args.post_update, APP_VERSION)
            release_metadata = load_release_metadata(args.post_update)
            from utils.gui import show_post_update_notice

            show_post_update_notice(release_metadata)
        except UpdateError as exc:
            logger.error("Post-update health confirmation failed [%s]: %s", exc.code, exc.user_message)
            _report_invalid_backend_configuration(exc)
            return 8
        except Exception as exc:
            logger.exception("Post-update health confirmation failed unexpectedly")
            _report_invalid_backend_configuration(exc)
            return 8
    elif args.rollback_notice:
        from utils.update_helper import rollback_notice

        try:
            message = rollback_notice(args.rollback_notice)
        except Exception:
            logger.exception("Failed to load rollback notice")
            message = "The previous CtrlSpeak version was restored after an update did not start safely."
        notify(message, title="CtrlSpeak update rolled back")
    run_tray()
    logger.info("CtrlSpeak shutting down cleanly")
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    atexit.register(release_single_instance_lock)
    atexit.register(shutdown_all)
    sys.exit(main(sys.argv))
