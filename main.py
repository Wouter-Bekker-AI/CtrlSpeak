# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
import atexit
import time

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
)

from utils.gui import show_splash_screen, ensure_mode_selected
from utils.models import initialize_transcriber


def main(argv: list[str]) -> int:
    args = parse_cli_args(argv)

    if args.uninstall:
        from utils.system import initiate_self_uninstall
        initiate_self_uninstall(None)
        return 0

    if args.force_sendinput:
        from utils.system import set_force_sendinput
        set_force_sendinput(True)

    # Single instance
    if not acquire_single_instance_lock():
        notify("CtrlSpeak is already running.")
        return 0

    # Load settings early
    load_settings()

    # CLI: offline transcription of a file
    if args.transcribe:
        return transcribe_cli(args.transcribe)

    # Splash
    show_splash_screen(SPLASH_DURATION_MS)

    # Ensure mode selected (client or client_server)
    ensure_mode_selected()

    # Start discovery listener for client mode visibility
    start_discovery_listener()

    # Orchestrate engine/server as needed based on mode
    with settings_lock:
        mode = settings.get("mode")

    if mode == "client_server":
        initialize_transcriber(interactive=False)   # warm-up local model when assets are ready
        start_server()
    elif mode == "client":
        time.sleep(1.0)  # small delay so discovery has time to populate

    run_tray()
    return 0


if __name__ == "__main__":
    atexit.register(release_single_instance_lock)
    atexit.register(shutdown_all)
    sys.exit(main(sys.argv))
