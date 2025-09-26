# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
import atexit
import time

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
    CLIENT_ONLY_BUILD,
    start_server,
    run_tray,
    apply_auto_setup,
)

from utils.gui import show_splash_screen, ensure_mode_selected, ensure_management_ui_thread
from utils.models import (
    initialize_transcriber,
    ensure_model_ready_for_local_server,
    ensure_initial_model_installation,
)

logger = get_logger(__name__)


def main(argv: list[str]) -> int:
    logger.info("CtrlSpeak starting up (version %s)", APP_VERSION)
    args = parse_cli_args(argv)
    logger.debug("Parsed CLI arguments: %s", args)

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

    if getattr(args, "cuda_only", False):
        from utils.models import (
            ensure_cuda_runtime_from_existing,
            install_cuda_runtime_with_progress,
            cuda_driver_available,
        )

        if not cuda_driver_available():
            logger.error("CUDA setup requested but no CUDA-capable GPU was detected on this system.")
            try:
                print("CUDA setup aborted: no CUDA-capable GPU detected.", file=sys.stderr)
            except Exception:
                logger.debug("Failed to write CUDA hardware warning to stderr", exc_info=True)
            return 1

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

    # Ensure mode selected (client or client_server)
    logger.debug("Ensuring operating mode is selected")
    ensure_mode_selected()

    # Auto-install the default speech model on first launch
    if not ensure_initial_model_installation():
        logger.error("Initial model installation failed or was aborted")
        return 0

    # Determine the selected mode now that setup is complete
    with settings_lock:
        mode = settings.get("mode")
    logger.info("CtrlSpeak running in '%s' mode", mode)

    # Automatically prepare local transcription assets when running the server locally
    if mode == "client_server":
        logger.info("Preparing local transcription assets for server mode")
        if not ensure_model_ready_for_local_server():
            logger.error("Failed to prepare local model for server mode")
            return 0

    # Start discovery listener for client mode visibility
    logger.debug("Starting discovery listener")
    start_discovery_listener()

    if mode == "client_server":
        logger.info("Initializing transcriber in background for warm-up")
        initialize_transcriber(interactive=False)   # warm-up local model when assets are ready
        logger.info("Starting local transcription server")
        start_server()
    elif mode == "client":
        logger.debug("Client mode selected; allowing discovery broadcast to populate")
        time.sleep(1.0)  # small delay so discovery has time to populate

    logger.info("Launching system tray UI")
    run_tray()
    logger.info("CtrlSpeak shutting down cleanly")
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    atexit.register(release_single_instance_lock)
    atexit.register(shutdown_all)
    sys.exit(main(sys.argv))
