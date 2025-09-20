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
    run_tray,
)
from utils.gui import show_splash_screen, ensure_mode_selected
from utils.models import initialize_transcriber
from utils.system import load_settings, parse_cli_args, transcribe_cli, CLIENT_ONLY_BUILD, start_server


def main(argv: list[str]) -> int:
    args = parse_cli_args(argv)

    if args.uninstall:
        from utils.system import initiate_self_uninstall
        initiate_self_uninstall(None)
        return 0

    if args.force_sendinput:
        from utils.system import set_force_sendinput
        set_force_sendinput(True)

    load_settings()

    if args.auto_setup:
        from utils.system import apply_auto_setup
        apply_auto_setup(args.auto_setup)

    cli_mode = args.transcribe is not None

    if not acquire_single_instance_lock():
        message = "CtrlSpeak is already running. Please close the existing instance before starting a new one."
        if cli_mode:
            print(message, file=sys.stderr)
        else:
            notify(message)
        return 0

    if not cli_mode:
        show_splash_screen(SPLASH_DURATION_MS)

    ensure_mode_selected()

    if cli_mode:
        start_discovery_listener()
        with settings_lock:
            mode = settings.get("mode")
        if mode == "client_server":
            initialize_transcriber()
        elif mode == "client":
            time.sleep(1.0)
        return transcribe_cli(args.transcribe)

    # Normal (tray) run
    start_discovery_listener()
    with settings_lock:
        mode = settings.get("mode")
    if mode == "client_server":
        initialize_transcriber()
        start_server()
    elif mode == "client":
        time.sleep(1.0)

    run_tray()
    return 0


if __name__ == "__main__":
    atexit.register(release_single_instance_lock)
    atexit.register(shutdown_all)
    sys.exit(main(sys.argv))
