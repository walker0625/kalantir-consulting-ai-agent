from pathlib import Path

_PATH_FILE = Path(__file__).resolve()

PROJECT_ROOT = _PATH_FILE.parent.parent

# === Root 하위 디렉토리 ===
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UTIL_DIR = PROJECT_ROOT / "util"
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
REPORT_DIR = PROJECT_ROOT / "report"
LOG_DIR = PROJECT_ROOT / "log"

# === Backend 하위 디렉토리 ===
ANALYSIS_DIR = BACKEND_DIR / "analysis"
DATA_DIR = BACKEND_DIR / "raw_data"
PROMPT_DIR = BACKEND_DIR / "prompt"

# === RawData 하위 디렉토리 ===
PDF_DIR = RAW_DATA_DIR / "pdf"