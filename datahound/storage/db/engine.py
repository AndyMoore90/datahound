"""Database engine scaffolding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class DatabaseConfig:
    """Connection configuration used by the DAL."""

    url: str
    echo: bool = False
    pool_size: int = 5
    pool_timeout: int = 30


class EngineFactory(Protocol):
    """Protocol describing how engines/sessions are produced."""

    def create_engine(self, config: DatabaseConfig):  # -> sqlalchemy.Engine
        ...

    def create_session(self):  # -> sqlalchemy.orm.Session
        ...
