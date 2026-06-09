"""Project configuration and default parameters."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_URL = "https://www.metoffice.gov.uk/hadobs/crutem3/data/CRUTEM3.nc"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = RAW_DATA_DIR / "CRUTEM3.nc"

# Leave as None to let the loader/preprocessor infer the anomaly variable.
TEMPERATURE_VARIABLE = None

START_YEAR = 1950
END_YEAR = 2000
MAX_MISSING_FRACTION = 0.25
EDGE_DENSITY = 0.01
DETREND = False
RANDOM_SEED = 42

