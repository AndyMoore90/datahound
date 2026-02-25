from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = PROJECT_ROOT / "logging"

TRANSCRIPT_PIPELINE = "transcript_pipeline"
SCHEDULER = "scheduler"
EVENT_DETECTION = "event_detection"
EXTRACTION = "extraction"
PIPELINE = "pipeline"
UPSERT = "upsert"
PERMITS = "permits"


def get_log_dir(process: str, company: str | None = None) -> Path:
    base = LOG_ROOT / process
    if company:
        base = base / company.replace(" ", "_").replace("/", "_")
    base.mkdir(parents=True, exist_ok=True)
    return base


def transcript_pipeline_dir() -> Path:
    return get_log_dir(TRANSCRIPT_PIPELINE)


def scheduler_dir() -> Path:
    return get_log_dir(SCHEDULER)


def event_detection_dir(company: str | None = None) -> Path:
    return get_log_dir(EVENT_DETECTION, company)


def event_detection_historical_dir(rule_name: str, company: str | None = None) -> Path:
    base = LOG_ROOT / "event_detection_historical"
    if company:
        base = base / company.replace(" ", "_").replace("/", "_")
    result = base / rule_name.replace(" ", "_").replace("/", "_").lower()
    result.mkdir(parents=True, exist_ok=True)
    return result


def extraction_dir(company: str | None = None) -> Path:
    return get_log_dir(EXTRACTION, company)


def pipeline_dir(company: str) -> Path:
    return get_log_dir(PIPELINE, company)


def upsert_changes_dir(company: str) -> Path:
    return get_log_dir("upsert_changes", company)


def permits_dir(company: str | None = None) -> Path:
    return get_log_dir(PERMITS, company)
