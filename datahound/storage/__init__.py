"""Storage subsystem scaffold for Phase 0.

This package currently exposes typed interfaces only. Concrete persistence
implementations will be provided in later refactor phases.
"""
from __future__ import annotations

from .dal import StorageDAL
from .manifest import DatasetManifest, ManifestEntry, ManifestStatus, RunManifest
from .schemas import DatasetSchema, SchemaField, SchemaRegistry

__all__ = [
    "StorageDAL",
    "DatasetManifest",
    "ManifestEntry",
    "ManifestStatus",
    "RunManifest",
    "DatasetSchema",
    "SchemaField",
    "SchemaRegistry",
]
