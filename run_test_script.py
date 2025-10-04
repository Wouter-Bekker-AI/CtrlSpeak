import sys
import time
import subprocess
import requests
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from utils.bot_integration import run_bot_test

TEST_WAV = project_root / "assets" / "test_16k_mono.wav"
SERVER_BASE = "http://127.0.0.1:65432"
TRANSCRIBE_URL = f"{SERVER_BASE}/transcribe"
HEALTH_URL = f"{SERVER_BASE}/ping"

main_path = project_root / "main.py"
cmd = [sys.executable, str(main_path), "--start-server-only"]
print(f"Starting CtrlSpeak server in background: {' '.join(cmd)}")
process = subprocess.Popen(cmd, cwd=str(project_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("Waiting for CtrlSpeak server to report healthy...")
start_time = time.time()
while time.time() - start_time < 90:
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        if response.ok:
            break
    except requests.RequestException:
        time.sleep(1)
else:
    print("Server did not become ready within timeout. Capturing logs...")
    stdout, stderr = process.communicate(timeout=5)
    print(stdout.decode(errors='ignore'))
    print(stderr.decode(errors='ignore'))
    raise SystemExit("Server failed to start")

if not TEST_WAV.is_file():
    raise SystemExit(f"Test WAV not found: {TEST_WAV}")

print("Server is ready. Running remote bot test...")
response_text = run_bot_test(str(TEST_WAV), stt_url=TRANSCRIBE_URL, identity="vision")
print(f"Bot's LLM Response: {response_text}")

print("Requesting server shutdown via /kill...")
try:
    requests.get(f"{SERVER_BASE}/kill", timeout=5)
except requests.RequestException as exc:
    print(f"Kill request failed: {exc}")

print("Waiting for CtrlSpeak background process to terminate...")
try:
    process.wait(timeout=30)
except subprocess.TimeoutExpired:
    print("Process did not exit promptly; terminating forcefully.")
    process.kill()

stdout, stderr = process.communicate(timeout=5)
if stdout:
    print("-- server stdout --")
    print(stdout.decode(errors='ignore'))
if stderr:
    print("-- server stderr --")
    print(stderr.decode(errors='ignore'))
