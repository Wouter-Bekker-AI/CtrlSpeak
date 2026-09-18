"""Versioned CtrlSpeak command adapter, deliberately outside Hermes core."""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path
from urllib.parse import urlsplit

import requests
from app.audio_transport import RELEASE_VERSION, prepare_audio, self_test
from app.stt_response import write_result


def transcribe(input_path: Path, language: str, output_path: Path, *, session=None):
    base = os.environ.get("CTRLSPEAK_GATEWAY_URL", "http://127.0.0.1:8765").rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("invalid gateway URL")
    headers = {}
    token = os.environ.get("CTRLSPEAK_CLIENT_TOKEN", "").strip()
    if token:
        headers["Authorization"] = "Bearer " + token
    key_path = os.environ.get("CTRLSPEAK_OPENAI_KEY_FILE", "")
    if key_path:
        with Path(key_path).open("r", encoding="utf-8") as handle:
            key = handle.read(4097).strip()
        if not key or len(key) > 4096 or any(c.isspace() for c in key):
            raise ValueError("invalid OpenAI key file")
        headers["X-CtrlSpeak-OpenAI-Key"] = key
    data = {"strategy": "ubuntu-gpu-preferred", "cleanup": "true"}
    if language:
        data["allowed_languages"] = language
    owned_session = session is None
    session = session or requests.Session()
    try:
        with prepare_audio(input_path) as audio, audio.path.open("rb") as source:
            logging.getLogger(__name__).info("Audio transport: %s", audio.telemetry())
            with session.post(base + "/v1/transcribe", headers=headers, data=data,
                              files={"audio": (audio.filename, source, audio.content_type)},
                              timeout=(5, 270), allow_redirects=False, stream=True) as response:
                if response.status_code != 200:
                    raise RuntimeError("CtrlSpeak returned HTTP " + str(response.status_code))
                content = bytearray()
                for chunk in response.iter_content(65536):
                    content.extend(chunk)
                    if len(content) > 2 * 1024 * 1024:
                        raise ValueError("CtrlSpeak response exceeds size limit")
                payload = json.loads(content)
                if not isinstance(payload, dict):
                    raise ValueError("invalid CtrlSpeak response")
                write_result(payload, output_path)
    finally:
        if owned_session:
            session.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=RELEASE_VERSION)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("input_path", nargs="?")
    parser.add_argument("language", nargs="?", default="en")
    parser.add_argument("output_path", nargs="?")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return 0
    if not args.input_path or not args.output_path:
        parser.error("input_path language output_path are required")
    os.umask(0o077)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def deadline(_signum, _frame):
        raise InterruptedError("STT helper deadline exceeded")

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(290)
    signal.signal(signal.SIGTERM, deadline)
    try:
        transcribe(Path(args.input_path), args.language, Path(args.output_path))
        return 0
    except Exception as exc:
        print("CtrlSpeak STT failed (" + type(exc).__name__ + ")", file=sys.stderr)
        return 2
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


if __name__ == "__main__":
    raise SystemExit(main())
