"""
Registry client. Handles install / uninstall / list / search across two sources:

 - Local file path or directory (primary supported path today — use this for distribution
   of packages bundled with code, or packages the user downloaded manually).
 - HTTP(S) URL to a tarball (downloaded, extracted, validated, installed).

The hosted registry server itself is out of scope for this repo. Users can host
their own and point TALKDB_REGISTRY_URL at it; the client will fetch package
tarballs and call the local install flow.
"""

from __future__ import annotations

import io
import logging
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from talkdb.config.settings import Settings
from talkdb.registry.index import InstalledPackage, PackageIndex
from talkdb.registry.package import SemanticPackage

_log = logging.getLogger("talkdb.registry")


class RegistryError(Exception):
    pass


@dataclass
class PackageSummary:
    name: str
    version: str
    description: str
    schema_type: str
    tables_covered: list[str]
    example_count: int
    verified: bool


class RegistryClient:
    def __init__(self, settings: Settings, index: PackageIndex | None = None):
        self.settings = settings
        self.packages_dir = Path(settings.registry_packages_path).resolve()
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        self.index = index or PackageIndex()

    # ----- Install -----

    def install(self, source: str) -> InstalledPackage:
        """
        Install a package from either a local directory/tarball or a URL.

        - If `source` is an existing local directory: copy it into packages_dir and index it.
        - If `source` is an existing local .tar.gz: extract and install.
        - If `source` looks like a URL: download + extract + install.
        - Otherwise: treat as a package name and try to fetch from the configured registry URL.
        """
        path = Path(source)
        if path.is_dir():
            return self._install_from_dir(path)
        if path.is_file() and path.suffixes[-2:] in (
            [".tar", ".gz"],
            [".tgz"],  # though this is [".tgz"] — kept for clarity
        ):
            return self._install_from_tarball(path)
        if source.startswith(("http://", "https://")):
            return self._install_from_url(source)
        # Fallback: treat as name and fetch from registry.
        return self._install_from_registry(source)

    def _install_from_dir(self, src: Path) -> InstalledPackage:
        # Validate first so we fail before copying a broken package.
        package = SemanticPackage.load(src)
        dest = self.packages_dir / package.manifest.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return self._register(package, installed_path=dest)

    def _install_from_tarball(self, tar_path: Path) -> InstalledPackage:
        with tarfile.open(tar_path, "r:gz") as tf:
            return self._install_from_tar_stream(tf)

    def _install_from_url(self, url: str) -> InstalledPackage:
        _log.info("fetching %s", url)
        with httpx.Client(timeout=30) as client:
            r = client.get(url, follow_redirects=True)
            r.raise_for_status()
            with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
                return self._install_from_tar_stream(tf)

    def _install_from_registry(self, package_name: str) -> InstalledPackage:
        if not self.settings.registry_url:
            raise RegistryError(
                f"No local path found for '{package_name}' and no registry URL configured."
            )
        url = f"{self.settings.registry_url.rstrip('/')}/api/v1/packages/{package_name}/download"
        return self._install_from_url(url)

    def _install_from_tar_stream(self, tf: tarfile.TarFile) -> InstalledPackage:
        """
        Safely extract a tarball into a temp dir, validate as a package, then promote.

        Security: reject member paths that would escape the extraction root
        (absolute paths or '..' components). Standard zip-slip guard.
        """
        staging = self.packages_dir / ".staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        for member in tf.getmembers():
            if member.isdev() or member.issym() or member.islnk():
                raise RegistryError(f"refusing to extract non-regular file: {member.name}")
            target = (staging / member.name).resolve()
            if not str(target).startswith(str(staging.resolve())):
                raise RegistryError(f"tarball entry escapes extraction root: {member.name}")
        tf.extractall(staging, filter="data")

        # The tarball may contain one top-level directory. Find it.
        entries = [p for p in staging.iterdir() if p.is_dir()]
        if len(entries) != 1:
            raise RegistryError(
                f"expected exactly one top-level directory in tarball, got {len(entries)}"
            )
        src = entries[0]
        package = SemanticPackage.load(src)
        dest = self.packages_dir / package.manifest.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        shutil.rmtree(staging, ignore_errors=True)
        return self._register(package, installed_path=dest)

    def _register(self, package: SemanticPackage, *, installed_path: Path) -> InstalledPackage:
        return self.index.add(
            name=package.manifest.name,
            version=package.manifest.version,
            source_path=str(installed_path),
            schema_type=package.manifest.schema_type,
            example_count=len(package.all_examples),
        )

    # ----- Uninstall -----

    def uninstall(self, name: str) -> bool:
        pkg = self.index.get(name)
        if pkg is None:
            return False
        # Delete the on-disk package if it's under our packages directory.
        path = Path(pkg.source_path)
        try:
            if path.is_dir() and path.resolve().is_relative_to(self.packages_dir.resolve()):
                shutil.rmtree(path)
        except Exception as e:  # noqa: BLE001
            _log.warning("uninstall: failed to delete %s: %s", path, e)
        return self.index.remove(name)

    # ----- List / search -----

    def list_installed(self) -> list[InstalledPackage]:
        return self.index.list()

    def load_all_installed(self) -> list[SemanticPackage]:
        """Load every installed package's semantic model into memory."""
        packages: list[SemanticPackage] = []
        for entry in self.index.list():
            try:
                packages.append(SemanticPackage.load(entry.source_path))
            except Exception as e:  # noqa: BLE001
                _log.warning("failed to load installed package %s: %s", entry.name, e)
        return packages

    def search(self, query: str) -> list[PackageSummary]:
        """Search a remote registry if configured; otherwise search installed packages."""
        if self.settings.registry_url:
            try:
                return self._search_remote(query)
            except Exception as e:  # noqa: BLE001
                _log.warning("remote search failed, falling back to local: %s", e)
        return self._search_local(query)

    def _search_remote(self, query: str) -> list[PackageSummary]:
        url = f"{self.settings.registry_url.rstrip('/')}/api/v1/packages"
        with httpx.Client(timeout=10) as client:
            r = client.get(url, params={"q": query})
            r.raise_for_status()
            data = r.json()
        return [PackageSummary(**entry) for entry in data.get("packages", [])]

    def _search_local(self, query: str) -> list[PackageSummary]:
        ql = query.lower()
        out: list[PackageSummary] = []
        for p in self.load_all_installed():
            m = p.manifest
            haystack = f"{m.name} {m.description} {m.schema_type} {' '.join(m.tables_covered)}".lower()
            if ql in haystack:
                out.append(
                    PackageSummary(
                        name=m.name,
                        version=m.version,
                        description=m.description,
                        schema_type=m.schema_type,
                        tables_covered=m.tables_covered,
                        example_count=m.example_count,
                        verified=m.verified,
                    )
                )
        return out
