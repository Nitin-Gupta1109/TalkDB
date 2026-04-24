"""Phase 8 unit tests: package loader, index, client (local-only paths)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from talkdb.config.settings import Settings
from talkdb.registry.client import RegistryClient
from talkdb.registry.index import PackageIndex
from talkdb.registry.package import PackageManifest, SemanticPackage


@pytest.fixture
def sample_pkg_dir(tmp_path: Path) -> Path:
    """Build a minimal valid package on disk for testing."""
    root = tmp_path / "my-pkg"
    (root / "examples").mkdir(parents=True)
    (root / "manifest.yaml").write_text(yaml.safe_dump({
        "name": "my-pkg",
        "version": "0.1.0",
        "description": "Test package",
        "schema_type": "custom",
    }))
    (root / "semantic_model.yaml").write_text(yaml.safe_dump({
        "version": "1.0",
        "metrics": [
            {"name": "foo", "description": "Test metric", "calculation": "COUNT(*)", "table": "t"}
        ],
        "examples": [
            {"question": "How many?", "sql": "SELECT COUNT(*) FROM t"}
        ],
    }))
    (root / "examples" / "queries.yaml").write_text(yaml.safe_dump({
        "examples": [{"question": "Alt question", "sql": "SELECT 1 FROM t"}]
    }))
    return root


class TestPackageManifest:
    def test_valid(self):
        m = PackageManifest(name="foo-bar", version="1.2.3", description="")
        assert m.name == "foo-bar"

    def test_bad_name(self):
        with pytest.raises(ValueError):
            PackageManifest(name="Foo!", version="1.0.0")

    def test_bad_version(self):
        with pytest.raises(ValueError):
            PackageManifest(name="foo", version="latest")


class TestSemanticPackage:
    def test_load_ok(self, sample_pkg_dir: Path):
        pkg = SemanticPackage.load(sample_pkg_dir)
        assert pkg.manifest.name == "my-pkg"
        assert len(pkg.semantic_model.metrics) == 1
        # 1 example in semantic_model.yaml + 1 in examples/queries.yaml
        assert len(pkg.all_examples) == 2

    def test_missing_manifest(self, tmp_path: Path):
        (tmp_path / "semantic_model.yaml").write_text("version: '1.0'\nmetrics: []")
        with pytest.raises(ValueError, match="manifest"):
            SemanticPackage.load(tmp_path)

    def test_missing_semantic_model(self, tmp_path: Path):
        (tmp_path / "manifest.yaml").write_text("name: x\nversion: '0.1.0'")
        with pytest.raises(ValueError, match="semantic_model"):
            SemanticPackage.load(tmp_path)

    def test_not_a_dir(self, tmp_path: Path):
        with pytest.raises(ValueError):
            SemanticPackage.load(tmp_path / "does-not-exist")


class TestPackageIndex:
    def test_add_list_remove(self, tmp_path: Path):
        idx = PackageIndex(path=str(tmp_path / "r.sqlite"))
        idx.add(name="x", version="1.0.0", source_path="/a", schema_type="custom", example_count=3)
        assert len(idx.list()) == 1
        assert idx.get("x").example_count == 3
        assert idx.remove("x")
        assert not idx.remove("x")
        assert len(idx.list()) == 0

    def test_upsert(self, tmp_path: Path):
        idx = PackageIndex(path=str(tmp_path / "r.sqlite"))
        idx.add(name="x", version="1.0.0", source_path="/a", schema_type="s", example_count=3)
        idx.add(name="x", version="1.1.0", source_path="/b", schema_type="s", example_count=5)
        rows = idx.list()
        assert len(rows) == 1
        assert rows[0].version == "1.1.0"
        assert rows[0].example_count == 5


class TestRegistryClient:
    def _client(self, tmp_path: Path) -> RegistryClient:
        settings = Settings(registry_packages_path=str(tmp_path / "pkgs"))
        idx = PackageIndex(path=str(tmp_path / "r.sqlite"))
        return RegistryClient(settings, index=idx)

    def test_install_from_dir(self, tmp_path: Path, sample_pkg_dir: Path):
        client = self._client(tmp_path)
        installed = client.install(str(sample_pkg_dir))
        assert installed.name == "my-pkg"
        assert installed.example_count == 2
        # Package was copied into packages_dir
        assert Path(installed.source_path).exists()

    def test_install_validates_and_rejects_broken(self, tmp_path: Path):
        bad = tmp_path / "broken"
        bad.mkdir()
        (bad / "manifest.yaml").write_text("name: bad\nversion: bad-semver")
        client = self._client(tmp_path)
        with pytest.raises(ValueError):
            client.install(str(bad))

    def test_uninstall(self, tmp_path: Path, sample_pkg_dir: Path):
        client = self._client(tmp_path)
        client.install(str(sample_pkg_dir))
        assert client.uninstall("my-pkg")
        assert len(client.list_installed()) == 0
        # Package directory was removed too
        assert not (client.packages_dir / "my-pkg").exists()

    def test_search_local(self, tmp_path: Path, sample_pkg_dir: Path):
        client = self._client(tmp_path)
        # Disable remote search so this test is hermetic.
        client.settings.registry_url = ""
        client.install(str(sample_pkg_dir))
        hits = client.search("test")
        assert len(hits) == 1
        assert hits[0].name == "my-pkg"

    def test_load_all_installed(self, tmp_path: Path, sample_pkg_dir: Path):
        client = self._client(tmp_path)
        client.install(str(sample_pkg_dir))
        pkgs = client.load_all_installed()
        assert len(pkgs) == 1
        assert pkgs[0].manifest.name == "my-pkg"
