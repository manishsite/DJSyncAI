import os
import sys
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "Logs"
LOG_FILE = LOGS_DIR / "logs.txt"

# Ensure Logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Load .env if present
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# App configuration
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

def append_to_logfile(text: str) -> None:
    """Safely append a timestamped entry to Logs/logs.txt and flush to disk."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8", buffering=1) as f:
            f.write(text.rstrip("\r\n") + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    except (PermissionError, OSError):
        pass
    except Exception as e:
        sys.stderr.write(f"Failed to write to {LOG_FILE}: {e}\n")

def log_event(level: str, message: str) -> None:
    """Format and record a log event."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{now}] [{level.upper()}] {message}"
    print(formatted, flush=True)
    append_to_logfile(formatted)
