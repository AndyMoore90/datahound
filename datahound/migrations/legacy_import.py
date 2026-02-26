"""CLI utility for bootstrapping legacy master data into canonical parquet layout.

This module scans a Windows-hosted (or mounted) directory that contains legacy
CSV/Parquet exports, validates each dataset against the configured schema, and
writes canonical parquet artifacts under ``companies/<company>/parquet``. Each
import registers dataset version metadata plus a synthetic pipeline run log and
produces a JSON migration report for auditability.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from datahound.download.types import DownloadConfig, load_config
from datahound.prepare.engine import read_master_columns_any, reorder_to_master
from datahound.storage.bootstrap import get_storage_dal_from_env
from datahound.storage.manifest import DatasetManifest, ManifestEntry, RunManifest
from datahound.storage.db.models import DatasetVersionRecord, PipelineRunRecord
from central_logging.config import pipeline_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_EXT = ".parquet"


@dataclass(slots=True)
class DatasetSpec:
    """Describes one dataset that should be imported."""

    dataset_type: str
    dataset_name: str
    master_filename: str
    schema_version: str
    expected_columns: Sequence[str] | None = None


@dataclass(slots=True)
class DatasetImportResult:
    """Per-dataset result that ends up in the migration report."""

    dataset_type: str
    dataset_name: str
    status: str
    schema_version: str | None
    source_file: Path | None = None
    output_file: Path | None = None
    row_count: int | None = None
    checksum: str | None = None
    version_tag: str | None = None
    validation_issues: List[str] = field(default_factory=list)
    skipped_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "dataset_type": self.dataset_type,
            "dataset_name": self.dataset_name,
            "status": self.status,
            "schema_version": self.schema_version,
        }
        if self.source_file:
            payload["source_file"] = str(self.source_file)
        if self.output_file:
            payload["output_file"] = str(self.output_file)
        if self.row_count is not None:
            payload["row_count"] = self.row_count
        if self.checksum:
            payload["checksum"] = self.checksum
        if self.version_tag:
            payload["version_tag"] = self.version_tag
        if self.validation_issues:
            payload["validation_issues"] = self.validation_issues
        if self.skipped_reason:
            payload["skipped_reason"] = self.skipped_reason
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(slots=True)
class MigrationReport:
    """Summary emitted at the end of a migration run."""

    company: str
    run_id: str
    source_path: Path
    output_dir: Path
    started_at: datetime
    finished_at: datetime
    results: Sequence[DatasetImportResult]
    dry_run: bool
    report_path: Path

    def success(self) -> bool:
        return all(r.status not in {"error", "validation_failed"} for r in self.results)

    def errors(self) -> List[str]:
        errs: List[str] = []
        for res in self.results:
            if res.error:
                errs.append(f"{res.dataset_name}: {res.error}")
            if res.status == "validation_failed":
                detail = "; ".join(res.validation_issues)
                if detail:
                    errs.append(f"{res.dataset_name}: schema mismatch ({detail})")
                else:
                    errs.append(f"{res.dataset_name}: schema mismatch")
        return errs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company": self.company,
            "run_id": self.run_id,
            "source_path": str(self.source_path),
            "output_dir": str(self.output_dir),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "dry_run": self.dry_run,
            "results": [res.to_dict() for res in self.results],
            "success": self.success(),
            "errors": self.errors(),
        }


def _has_windows_drive(text: str) -> bool:
    return len(text) >= 2 and text[1] == ":" and text[0].isalpha()


def _normalize_pathish(value: Path | str) -> Path:
    raw = str(value).strip()
    expanded = os.path.expandvars(raw)
    normalized = expanded.replace("\\", "/")
    return Path(normalized).expanduser()


def _resolve_path(value: Path | str, *, base: Path | None = PROJECT_ROOT) -> Path:
    normalized = _normalize_pathish(value)
    normalized_text = str(normalized)
    if normalized.is_absolute() or normalized_text.startswith("//") or _has_windows_drive(normalized_text):
        return normalized
    if base is None:
        return normalized.resolve()
    return (base / normalized).resolve()


def _slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "_")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


class LegacyDataImporter:
    """Coordinates legacy data imports."""

    def __init__(
        self,
        *,
        company: str,
        source_dir: Path,
        config_path: Path | None = None,
        datasets: Sequence[str] | None = None,
        recursive: bool = True,
        dry_run: bool = False,
        report_path: Path | None = None,
    ) -> None:
        self.company = company
        self.source_dir = _resolve_path(source_dir, base=None)
        self.config_path = _resolve_path(
            config_path or (PROJECT_ROOT / "companies" / company / "config.json")
        )
        self.datasets_filter = {d.strip().lower() for d in datasets} if datasets else None
        self.recursive = recursive
        self.dry_run = dry_run
        company_slug = _slugify(company)
        default_report = PROJECT_ROOT / "reports" / "migrations" / f"{company_slug}_migration.json"
        base_report = _resolve_path(report_path, base=None) if report_path else default_report
        suffix = base_report.suffix or ".json"
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        base_stem = base_report.stem
        self.report_path = base_report.with_name(f"{base_stem}_{timestamp}{suffix}")
        self.run_id = uuid.uuid4().hex[:12]
        self.pipeline_name = "legacy_data_import"
        self.stage = "bootstrap"
        self._dataset_specs: Dict[str, DatasetSpec] = {}
        self._known_versions: set[tuple[str, str]] = set()
        self._pipeline_log_dir = pipeline_dir(company)
        self._dataset_versions_path = self._pipeline_log_dir / "dataset_versions.jsonl"
        self._pipeline_runs_path = self._pipeline_log_dir / "pipeline_runs.jsonl"
        self._target_dir = PROJECT_ROOT / "companies" / company / "parquet"
        self._storage_dal = None if self.dry_run else get_storage_dal_from_env()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> MigrationReport:
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        config = self._load_company_config()
        self._dataset_specs = self._build_dataset_specs(config)
        selected_types = self._filter_dataset_types()
        source_map = self._discover_source_files(selected_types)
        self._load_existing_dataset_versions()
        self._target_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(UTC)
        self._start_pipeline_run(started)
        results: List[DatasetImportResult] = []
        for dataset_type in selected_types:
            spec = self._dataset_specs[dataset_type]
            result = self._process_dataset(spec, source_map.get(dataset_type))
            results.append(result)
        finished = datetime.now(UTC)
        report = MigrationReport(
            company=self.company,
            run_id=self.run_id,
            source_path=self.source_dir,
            output_dir=self._target_dir,
            started_at=started,
            finished_at=finished,
            results=tuple(results),
            dry_run=self.dry_run,
            report_path=self.report_path,
        )
        self._write_report(report)
        self._record_pipeline_run(report)
        self._finish_pipeline_run(report)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_pipeline_run(self, started_at: datetime) -> None:
        if self._storage_dal is None:
            return
        run = PipelineRunRecord(
            id=None,
            run_id=self.run_id,
            company=self.company,
            pipeline_name=self.pipeline_name,
            stage=self.stage,
            status="running",
            input_manifest_json=None,
            output_manifest_json=None,
            error_json=None,
            started_at=started_at,
            finished_at=None,
        )
        self._storage_dal.start_pipeline_run(run)

    def _finish_pipeline_run(self, report: MigrationReport) -> None:
        if self._storage_dal is None:
            return
        outputs: List[DatasetManifest] = []
        for res in report.results:
            if res.output_file and res.version_tag and res.checksum and res.row_count is not None:
                entry = ManifestEntry(
                    dataset_name=res.dataset_name,
                    version_tag=res.version_tag,
                    file_path=res.output_file,
                    schema_version=res.schema_version,
                    row_count=res.row_count,
                    checksum=res.checksum,
                )
                outputs.append(
                    DatasetManifest(
                        dataset_name=res.dataset_name,
                        entries=[entry],
                        produced_by_run_id=self.run_id,
                        created_at=report.finished_at,
                    )
                )
        status = "success" if report.success() else "failed"
        manifest = RunManifest(
            run_id=self.run_id,
            pipeline_name=self.pipeline_name,
            company=self.company,
            stage=self.stage,
            status=status,
            started_at=report.started_at,
            finished_at=report.finished_at,
            input_datasets=tuple(),
            output_datasets=tuple(outputs),
            metadata={
                "source_path": str(report.source_path),
                "dry_run": self.dry_run,
            },
            error={"messages": report.errors()} if not report.success() else None,
        )
        self._storage_dal.finish_pipeline_run(
            self.run_id,
            status=status,
            manifest=manifest,
            error=manifest.error,
        )

    def _load_company_config(self) -> DownloadConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Company config not found: {self.config_path}")
        cfg = load_config(self.config_path)
        cfg.data_dir = _resolve_path(cfg.data_dir)
        if cfg.prepare:
            cfg.prepare.tables_dir = _resolve_path(cfg.prepare.tables_dir)
        return cfg

    def _build_dataset_specs(self, config: DownloadConfig) -> Dict[str, DatasetSpec]:
        if not config.prepare:
            raise RuntimeError("Prepare configuration is required to determine schemas")
        specs: Dict[str, DatasetSpec] = {}
        for dataset_type, master_name in config.prepare.file_type_to_master.items():
            dataset_name = Path(master_name).stem or dataset_type.capitalize()
            schema_version = dataset_name
            try:
                expected = read_master_columns_any(config.prepare.tables_dir, master_name, prefer_parquet=True)
            except Exception:
                expected = None
            specs[dataset_type.lower()] = DatasetSpec(
                dataset_type=dataset_type.lower(),
                dataset_name=dataset_name,
                master_filename=master_name,
                schema_version=schema_version,
                expected_columns=tuple(expected) if expected else None,
            )
        return specs

    def _filter_dataset_types(self) -> List[str]:
        available = sorted(self._dataset_specs.keys())
        if not self.datasets_filter:
            return available
        filtered = [d for d in available if d in self.datasets_filter or self._dataset_specs[d].dataset_name.lower() in self.datasets_filter]
        if not filtered:
            raise ValueError("Datasets filter did not match any known types")
        return filtered

    def _discover_source_files(self, dataset_types: Iterable[str]) -> Dict[str, Path]:
        needles = {
            dt: self._tokens_for_dataset(self._dataset_specs[dt]) for dt in dataset_types
        }
        matches: Dict[str, List[Path]] = {dt: [] for dt in dataset_types}
        search_iter: Iterable[Path]
        if self.recursive:
            search_iter = chain(self.source_dir.rglob("*.parquet"), self.source_dir.rglob("*.csv"))
        else:
            search_iter = chain(self.source_dir.glob("*.parquet"), self.source_dir.glob("*.csv"))
        for path in search_iter:
            if not path.is_file():
                continue
            lower_name = path.name.lower()
            for dataset_type, tokens in needles.items():
                if any(token in lower_name for token in tokens):
                    matches[dataset_type].append(path)
                    break
        chosen: Dict[str, Path] = {}
        for dataset_type, paths in matches.items():
            if not paths:
                continue
            paths.sort(key=self._source_sort_key)
            chosen[dataset_type] = paths[0]
        return chosen

    def _tokens_for_dataset(self, spec: DatasetSpec) -> List[str]:
        tokens = {
            spec.dataset_type.lower(),
            spec.dataset_type.rstrip("s").lower(),
            spec.dataset_name.lower(),
            spec.dataset_name.lower().replace(" ", ""),
            spec.dataset_name.lower().replace(" ", "_"),
            spec.dataset_name.lower().replace("_", ""),
        }
        return [t for t in tokens if t]

    def _source_sort_key(self, path: Path) -> tuple[int, float]:
        suf_score = 0 if path.suffix.lower() == ".parquet" else 1
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0
        return (suf_score, -mtime)

    def _load_existing_dataset_versions(self) -> None:
        if not self._dataset_versions_path.exists():
            return
        try:
            for line in self._dataset_versions_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                key = (rec.get("dataset_name"), rec.get("version_tag"))
                if key[0] and key[1]:
                    self._known_versions.add((str(key[0]), str(key[1])))
        except Exception:
            pass

    def _process_dataset(self, spec: DatasetSpec, source_path: Optional[Path]) -> DatasetImportResult:
        result = DatasetImportResult(
            dataset_type=spec.dataset_type,
            dataset_name=spec.dataset_name,
            status="skipped_no_source",
            schema_version=spec.schema_version,
        )
        if source_path is None:
            result.skipped_reason = "no matching source file"
            return result
        result.source_file = source_path
        try:
            df = self._read_source_file(source_path)
            if spec.dataset_type == "calls" and "Call ID" not in df.columns and "ID" in df.columns:
                df["Call ID"] = df["ID"]
            issues = self._validate_schema(df, spec.expected_columns)
            if issues:
                result.status = "validation_failed"
                result.validation_issues = issues
                return result
            canonical_df = self._align_to_schema(df, spec.expected_columns)
            row_count = int(len(canonical_df))
            payload, checksum = self._to_parquet_bytes(canonical_df)
            version_tag = f"legacy-import-{checksum[:12]}"
            target_path = self._target_dir / f"{spec.dataset_name}{CANONICAL_EXT}"
            result.status = "imported"
            result.row_count = row_count
            result.checksum = checksum
            result.version_tag = version_tag
            result.output_file = target_path
            if self._has_dataset_version(spec.dataset_name, version_tag):
                result.status = "skipped_existing_version"
                result.skipped_reason = "dataset version already registered"
                return result
            if not self.dry_run and target_path.exists():
                existing_checksum = self._file_checksum(target_path)
                if existing_checksum == checksum:
                    if self._has_dataset_version(spec.dataset_name, version_tag):
                        result.status = "skipped_existing_version"
                        result.skipped_reason = "matching checksum already present"
                        return result
                    self._register_dataset_version(spec, version_tag, row_count, checksum, target_path)
                    result.status = "registered_existing_artifact"
                    result.skipped_reason = "file already matched; registered metadata"
                    return result
            if self.dry_run:
                result.status = "dry_run"
                return result
            self._write_parquet_file(target_path, payload)
            self._register_dataset_version(spec, version_tag, row_count, checksum, target_path)
            return result
        except Exception as exc:
            result.status = "error"
            result.error = str(exc)
            return result

    def _read_source_file(self, path: Path) -> pd.DataFrame:
        suf = path.suffix.lower()
        if suf == ".parquet":
            return pd.read_parquet(path)
        if suf == ".csv":
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False)
            except UnicodeDecodeError:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")
        raise ValueError(f"Unsupported file type: {path}")

    def _validate_schema(self, df: pd.DataFrame, expected: Sequence[str] | None) -> List[str]:
        if not expected:
            return []
        df_cols = [str(c) for c in df.columns]
        if df_cols == list(expected):
            return []
        missing = [c for c in expected if c not in df_cols]
        extra = [c for c in df_cols if c not in expected]
        issues: List[str] = []
        if missing:
            issues.append(f"missing columns: {missing}")
        if extra:
            issues.append(f"unexpected columns: {extra}")
        return issues

    def _align_to_schema(self, df: pd.DataFrame, expected: Sequence[str] | None) -> pd.DataFrame:
        aligned = df.copy()
        if expected:
            aligned = reorder_to_master(aligned, list(expected))
        for col in aligned.columns:
            aligned[col] = aligned[col].astype("string").fillna("")
        return aligned

    def _to_parquet_bytes(self, df: pd.DataFrame) -> tuple[bytes, str]:
        table = pa.Table.from_pandas(df, preserve_index=False)
        buf = BytesIO()
        pq.write_table(table, buf)
        payload = buf.getvalue()
        checksum = hashlib.sha256(payload).hexdigest()
        return payload, checksum

    def _file_checksum(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_parquet_file(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(payload)
        tmp.replace(path)

    def _has_dataset_version(self, dataset_name: str, version_tag: str) -> bool:
        return (dataset_name, version_tag) in self._known_versions

    def _register_dataset_version(
        self,
        spec: DatasetSpec,
        version_tag: str,
        row_count: int,
        checksum: str,
        target_path: Path,
    ) -> None:
        record = DatasetVersionRecord(
            id=None,
            dataset_name=spec.dataset_name,
            company=self.company,
            version_tag=version_tag,
            schema_version=spec.schema_version,
            file_path=target_path,
            row_count=row_count,
            checksum=checksum,
            produced_by_run_id=self.run_id,
            created_at=datetime.now(UTC),
        )
        payload = {
            "dataset_name": record.dataset_name,
            "company": record.company,
            "version_tag": record.version_tag,
            "schema_version": record.schema_version,
            "file_path": str(record.file_path),
            "row_count": record.row_count,
            "checksum": record.checksum,
            "produced_by_run_id": record.produced_by_run_id,
            "created_at": record.created_at.isoformat(),
        }
        self._append_jsonl(self._dataset_versions_path, payload)
        if self._storage_dal is not None:
            self._storage_dal.register_dataset_version(record)
        self._known_versions.add((record.dataset_name, version_tag))

    def _write_report(self, report: MigrationReport) -> None:
        report.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report.report_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)

    def _record_pipeline_run(self, report: MigrationReport) -> None:
        outputs: List[DatasetManifest] = []
        for res in report.results:
            if res.output_file and res.version_tag and res.checksum and res.row_count is not None:
                entry = ManifestEntry(
                    dataset_name=res.dataset_name,
                    version_tag=res.version_tag,
                    file_path=res.output_file,
                    schema_version=res.schema_version,
                    row_count=res.row_count,
                    checksum=res.checksum,
                )
                outputs.append(
                    DatasetManifest(
                        dataset_name=res.dataset_name,
                        entries=[entry],
                        produced_by_run_id=self.run_id,
                        created_at=report.finished_at,
                    )
                )
        status = "success" if report.success() else "failed"
        manifest = RunManifest(
            run_id=self.run_id,
            pipeline_name=self.pipeline_name,
            company=self.company,
            stage=self.stage,
            status=status,
            started_at=report.started_at,
            finished_at=report.finished_at,
            input_datasets=tuple(),
            output_datasets=tuple(outputs),
            metadata={
                "source_path": str(report.source_path),
                "dry_run": self.dry_run,
            },
            error={"messages": report.errors()} if not report.success() else None,
        )
        payload = {
            "run_id": manifest.run_id,
            "company": manifest.company,
            "pipeline_name": manifest.pipeline_name,
            "stage": manifest.stage,
            "status": manifest.status,
            "input_manifest_json": None,
            "output_manifest_json": manifest.as_dict(),
            "error_json": manifest.error,
            "started_at": manifest.started_at.isoformat(),
            "finished_at": manifest.finished_at.isoformat(),
        }
        self._append_jsonl(self._pipeline_runs_path, payload)

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(payload), default=_json_default) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap legacy CSV/Parquet files into canonical parquet datasets.")
    parser.add_argument("--company", required=True, help="Company name (matches companies/<name> directory)")
    parser.add_argument("--source", required=True, help="Path to legacy data directory (supports Windows paths)")
    parser.add_argument("--config", help="Optional path to company config.json (defaults to companies/<company>/config.json)")
    parser.add_argument("--datasets", nargs="*", help="Optional dataset types/names to limit the import")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan the top-level source directory")
    parser.add_argument("--dry-run", action="store_true", help="Run validations without writing files or metadata")
    parser.add_argument("--report", help="Optional path for the migration report JSON (default: reports/migrations)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    importer = LegacyDataImporter(
        company=args.company,
        source_dir=Path(args.source),
        config_path=Path(args.config) if args.config else None,
        datasets=args.datasets,
        recursive=not args.no_recursive,
        dry_run=bool(args.dry_run),
        report_path=Path(args.report) if args.report else None,
    )
    report = importer.run()
    status = "SUCCESS" if report.success() else "FAILED"
    print(f"[{status}] Imported {sum(1 for r in report.results if r.status == 'imported')} datasets for {report.company} (run_id={report.run_id})")
    print(f"Report: {report.report_path}")
    if report.errors():
        print("Issues:")
        for err in report.errors():
            print(f"  - {err}")
    return 0 if report.success() else 1


if __name__ == "__main__":
    raise SystemExit(main())
