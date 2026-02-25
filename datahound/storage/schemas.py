"""Dataset schema abstractions for curated tables."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence


@dataclass(slots=True)
class SchemaField:
    """Describes a single column inside a dataset schema."""

    name: str
    dtype: str
    nullable: bool = True
    description: str | None = None


@dataclass(slots=True)
class DatasetSchema:
    """Represents a named schema version for a dataset."""

    dataset_name: str
    version: str
    fields: Sequence[SchemaField]
    checksum: str | None = None
    metadata: Mapping[str, str] | None = None

    def field_names(self) -> Sequence[str]:
        return [field.name for field in self.fields]


class SchemaRegistry:
    """In-memory registry placeholder until a persistent catalog is wired up."""

    def __init__(self) -> None:
        self._schemas: Dict[tuple[str, str], DatasetSchema] = {}

    def register(self, schema: DatasetSchema) -> None:
        key = (schema.dataset_name, schema.version)
        self._schemas[key] = schema

    def get(self, dataset_name: str, version: str) -> DatasetSchema | None:
        return self._schemas.get((dataset_name, version))

    def latest(self, dataset_name: str) -> DatasetSchema | None:
        for (name, _version), schema in reversed(list(self._schemas.items())):
            if name == dataset_name:
                return schema
        return None

    def all(self) -> Iterable[DatasetSchema]:
        return self._schemas.values()
