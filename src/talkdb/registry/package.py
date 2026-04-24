"""
Community package format.

Per CLAUDE.md: "Community packages are YAML + examples, not code. Packages contain
semantic model definitions and proven query patterns. They never contain executable
code — security by design."

Each package is a directory with this structure:

    <name>/
    ├── manifest.yaml           # Package metadata
    ├── semantic_model.yaml     # Metrics, tables, joins
    ├── examples/
    │   └── queries.yaml        # Proven question -> SQL pairs (optional)
    └── README.md               # (optional)

`SemanticPackage.load(path)` validates both shape and contents. No code execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from talkdb.schema.semantic_model import SemanticExample, SemanticModel


class PackageManifest(BaseModel):
    """Contents of manifest.yaml."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    schema_type: str = "custom"
    compatible_dialects: list[str] = Field(default_factory=list)
    tables_covered: list[str] = Field(default_factory=list)
    example_count: int = 0
    verified: bool = False
    dependencies: list[str] = Field(default_factory=list)
    homepage: str | None = None
    license: str = "MIT"

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        # Package name must be a safe directory identifier. Lower-case, hyphen, underscore, digits.
        import re

        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", v):
            raise ValueError(
                "package name must be lower-case ASCII and contain only letters, digits, '-' or '_'"
            )
        return v

    @field_validator("version")
    @classmethod
    def _semver_like(cls, v: str) -> str:
        # Accept semver-like strings. Be permissive: don't parse into full semver.
        import re

        if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+.][A-Za-z0-9._-]+)?", v):
            raise ValueError("version must look like semver (e.g. 1.2.0)")
        return v


@dataclass
class SemanticPackage:
    """A loaded, validated community package. Holds its semantic model in memory."""

    manifest: PackageManifest
    semantic_model: SemanticModel
    examples: list[SemanticExample]  # Merged from examples/queries.yaml (extra, beyond semantic_model.examples)
    source_path: Path

    @classmethod
    def load(cls, path: str | Path) -> SemanticPackage:
        """
        Load and validate a package directory. Raises ValueError on any issue.

        Security: the loader only reads YAML. No code execution. Extra files in
        the package directory (beyond manifest/semantic_model/examples/README) are ignored.
        """
        base = Path(path).resolve()
        if not base.is_dir():
            raise ValueError(f"package path is not a directory: {base}")

        manifest_path = base / "manifest.yaml"
        if not manifest_path.exists():
            raise ValueError(f"missing manifest.yaml in {base}")

        with manifest_path.open() as f:
            manifest_data = yaml.safe_load(f) or {}
        manifest = PackageManifest.model_validate(manifest_data)

        model_path = base / "semantic_model.yaml"
        if not model_path.exists():
            raise ValueError(f"missing semantic_model.yaml in {base}")
        semantic_model = SemanticModel.load(model_path)

        extra_examples: list[SemanticExample] = []
        examples_path = base / "examples" / "queries.yaml"
        if examples_path.exists():
            with examples_path.open() as f:
                data = yaml.safe_load(f) or {}
            for entry in data.get("examples", []):
                extra_examples.append(SemanticExample.model_validate(entry))

        return cls(
            manifest=manifest,
            semantic_model=semantic_model,
            examples=extra_examples,
            source_path=base,
        )

    @property
    def all_examples(self) -> list[SemanticExample]:
        """Union of examples in the semantic model + the extra queries.yaml."""
        return [*self.semantic_model.examples, *self.examples]
