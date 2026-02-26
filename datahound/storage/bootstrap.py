"""Factory helpers for configuring the Storage DAL."""
from __future__ import annotations

import os
from functools import lru_cache

from .dal import DALDependencies, StorageDAL
from .dal_impl import SQLStorageDAL
from .db.engine import DatabaseConfig, SQLAlchemyEngineFactory
from .db.models import StorageBase
from .db.repos.event_sql_repo import SQLEventRepository
from .db.repos.notification_sql_repo import SQLNotificationRepository
from .db.repos.review_sql_repo import SQLReviewRepository
from .db.repos.run_sql_repo import SQLRunRepository
from .db.repos.scheduler_sql_repo import SQLSchedulerRepository


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_storage_dal_from_env() -> StorageDAL | None:
    """Instantiate a StorageDAL if ``DATAHOUND_STORAGE_URL`` is configured."""

    url = os.getenv("DATAHOUND_STORAGE_URL") or os.getenv("DATABASE_URL")
    if not url:
        return None

    try:
        config = DatabaseConfig(
            url=url,
            echo=_env_flag("DATAHOUND_STORAGE_ECHO"),
        )
        engine_factory = SQLAlchemyEngineFactory(config)
        engine = engine_factory.create_engine()
        StorageBase.metadata.create_all(engine)
        session_factory = engine_factory.create_session_factory()
        dependencies = DALDependencies(
            scheduler_repo=SQLSchedulerRepository(session_factory),
            run_repo=SQLRunRepository(session_factory),
            event_repo=SQLEventRepository(session_factory),
            notification_repo=SQLNotificationRepository(session_factory),
            review_repo=SQLReviewRepository(session_factory),
        )
        return SQLStorageDAL(dependencies)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[storage] Failed to initialize database DAL: {exc}")
        return None
