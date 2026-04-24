"""
Local index of installed packages.

SQLite-backed. Tracks which packages are installed, their versions, their disk location,
and install timestamps. Separate from the pattern store (which holds individual
query patterns from corrections).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import Column, DateTime, Integer, String, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Session


class _Base(DeclarativeBase):
    pass


class _PackageRow(_Base):
    __tablename__ = "installed_packages"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True)
    version = Column(String(100), nullable=False)
    source_path = Column(String(1000), nullable=False)
    schema_type = Column(String(100), nullable=True)
    installed_at = Column(DateTime, nullable=False, default=func.now())
    example_count = Column(Integer, nullable=False, default=0)


@dataclass
class InstalledPackage:
    name: str
    version: str
    source_path: str
    schema_type: str | None
    installed_at: datetime
    example_count: int


class PackageIndex:
    """SQLite-backed registry of installed packages."""

    def __init__(self, path: str = "./data/registry.sqlite"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{p}", future=True)
        _Base.metadata.create_all(self._engine)

    def add(
        self,
        *,
        name: str,
        version: str,
        source_path: str,
        schema_type: str | None,
        example_count: int,
    ) -> InstalledPackage:
        with Session(self._engine) as session:
            row = session.query(_PackageRow).filter_by(name=name).one_or_none()
            if row is None:
                row = _PackageRow(name=name)
                session.add(row)
            row.version = version
            row.source_path = source_path
            row.schema_type = schema_type
            row.example_count = example_count
            session.commit()
            session.refresh(row)
            return _to_dc(row)

    def remove(self, name: str) -> bool:
        with Session(self._engine) as session:
            row = session.query(_PackageRow).filter_by(name=name).one_or_none()
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def list(self) -> list[InstalledPackage]:
        with Session(self._engine) as session:
            rows = session.query(_PackageRow).order_by(_PackageRow.name).all()
            return [_to_dc(r) for r in rows]

    def get(self, name: str) -> InstalledPackage | None:
        with Session(self._engine) as session:
            row = session.query(_PackageRow).filter_by(name=name).one_or_none()
            return _to_dc(row) if row else None


def _to_dc(row: _PackageRow) -> InstalledPackage:
    return InstalledPackage(
        name=row.name,
        version=row.version,
        source_path=row.source_path,
        schema_type=row.schema_type,
        installed_at=row.installed_at,
        example_count=row.example_count,
    )
