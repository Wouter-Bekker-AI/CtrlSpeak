import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config_paths import get_logs_dir

log_file = get_logs_dir() / "ctrlspeak.log"
print(log_file)
