import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config_paths import get_config_dir

config_dir = get_config_dir()
log_file = config_dir / "logs" / "ctrlspeak.log"
print(log_file)
