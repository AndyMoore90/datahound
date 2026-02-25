"""Database engine scaffolding backed by SQLAlchemy."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Protocol, TypeAlias

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

# ``sessionmaker`` is callable and parameterisable; ``SessionFactory`` captures the
# configured callable we hand to the rest of the app without forcing SQLAlchemy
# usage on call sites yet.
SessionFactory: TypeAlias = sessionmaker[Session]


@dataclass(slots=True)
class DatabaseConfig:
    """Connection configuration used by the DAL."""

    url: str
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int | None = 1800
    pool_pre_ping: bool = True


class EngineFactory(Protocol):
    """Protocol describing how engines/sessions are produced."""

    def create_engine(self) -> Engine:
        ...

    def create_session_factory(self) -> SessionFactory:
        ...

    def create_session(self) -> Session:
        ...


def create_sqlalchemy_engine(config: DatabaseConfig) -> Engine:
    """Create a synchronous SQLAlchemy engine configured for Postgres."""

    return create_engine(
        config.url,
        echo=config.echo,
        pool_size=config.pool_size,
        pool_timeout=config.pool_timeout,
        max_overflow=config.max_overflow,
        pool_recycle=config.pool_recycle,
        pool_pre_ping=config.pool_pre_ping,
        future=True,
    )


def create_session_factory(engine: Engine) -> SessionFactory:
    """Produce the session factory bound to ``engine``."""

    return sessionmaker(bind=engine, class_=Session, autoflush=False, expire_on_commit=False)


class SQLAlchemyEngineFactory:
    """Lazily instantiates the engine and matching session factory."""

    def __init__(self, config: DatabaseConfig):
        self._config = config
        self._engine: Engine | None = None
        self._session_factory: SessionFactory | None = None

    def create_engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_sqlalchemy_engine(self._config)
        return self._engine

    def create_session_factory(self) -> SessionFactory:
        if self._session_factory is None:
            engine = self.create_engine()
            self._session_factory = create_session_factory(engine)
        return self._session_factory

    def create_session(self) -> Session:
        return self.create_session_factory()()


@contextmanager
def session_scope(session_factory: SessionFactory) -> Iterator[Session]:
    """Simple context manager for non-ORM callers that want scoped sessions."""

    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:  # pragma: no cover - defensive rollback guard
        session.rollback()
        raise
    finally:
        session.close()
