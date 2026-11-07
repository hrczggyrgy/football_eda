"""
Versioned feature contract for the predictive engine.

This module defines the canonical feature schema shared by training, validation,
and inference. Any change to feature names, order, dtypes, or missing-value
semantics requires a version bump.

The schema is derived from the actual fitted predictor names and order in
the persisted model bundle. It is the single source of truth for feature
compatibility between historical replay and prospective inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
import json
from pathlib import Path


# Current schema version. Increment when feature names, order, dtypes,
# or missing-value semantics change.
FEATURE_SCHEMA_VERSION = "1.0.0"

# Feature requirement levels
Requirement = Literal["required", "optional"]


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for a single predictor."""
    name: str
    dtype: str  # e.g., "float64", "int64"
    meaning: str
    availability: str  # When this feature becomes available relative to the fixture
    grouping: str  # Team/division/season policy
    missing_meaning: str  # What a null value means
    fallback: str  # How to handle missing at inference time
    units: str
    requirement: Requirement = "required"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "meaning": self.meaning,
            "availability": self.availability,
            "grouping": self.grouping,
            "missing_meaning": self.missing_meaning,
            "fallback": self.fallback,
            "units": self.units,
            "requirement": self.requirement,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        return cls(**d)


@dataclass
class FeatureSchema:
    """Complete versioned feature contract."""
    version: str
    features: List[FeatureSpec]
    class_order: List[str] = field(default_factory=lambda: ["H", "D", "A"])
    metadata: Dict = field(default_factory=dict)

    def feature_names(self) -> List[str]:
        return [f.name for f in self.features]

    def required_names(self) -> List[str]:
        return [f.name for f in self.features if f.requirement == "required"]

    def optional_names(self) -> List[str]:
        return [f.name for f in self.features if f.requirement == "optional"]

    def get_spec(self, name: str) -> Optional[FeatureSpec]:
        for f in self.features:
            if f.name == name:
                return f
        return None

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "features": [f.to_dict() for f in self.features],
            "class_order": self.class_order,
            "metadata": self.metadata,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "FeatureSchema":
        data = json.loads(path.read_text())
        return cls(
            version=data["version"],
            features=[FeatureSpec.from_dict(f) for f in data["features"]],
            class_order=data.get("class_order", ["H", "D", "A"]),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_fitted_bundle(cls, feature_names: List[str], version: str = FEATURE_SCHEMA_VERSION) -> "FeatureSchema":
        """Create a schema from the actual fitted feature names (best-effort specs)."""
        # These are best-effort specifications inferred from the feature names.
        # They should be reviewed and completed manually.
        specs = []
        for name in feature_names:
            spec = _infer_feature_spec(name)
            specs.append(spec)
        return cls(version=version, features=specs)


def _infer_feature_spec(name: str) -> FeatureSpec:
    """Infer a best-effort FeatureSpec from the feature name."""
    # Default values
    dtype = "float64"
    meaning = "Auto-generated from feature name"
    availability = "Pre-match (strictly lagged)"
    grouping = "Team/division/season"
    missing_meaning = "No observed history for this team/season/context"
    fallback = "Training-fitted column mean imputation"
    units = "Various (PPG, goals, counts, ratings, etc.)"
    requirement = "required"

    # Categorize by prefix/pattern
    if name.startswith("Home_") or name.startswith("Away_"):
        base = name[5:] if name.startswith("Home_") else name[5:]
        venue = "Home" if name.startswith("Home_") else "Away"
        meaning = f"{venue} team's {base.replace('_', ' ').lower()}"
    elif name in ("PPG_Gap", "GF_Cum_Gap", "GA_Cum_Gap", "Rest_Days_Diff"):
        meaning = f"Match-level difference: {name.replace('_', ' ').lower()}"
        grouping = "Match-level (home - away)"
    elif name in ("Elo_Exp_Home", "Elo_Rating_Diff"):
        meaning = f"Match-level Elo: {name.replace('_', ' ').lower()}"
        grouping = "Match-level"
    elif name.startswith("League_Avg_"):
        meaning = f"League scoring environment: {name.replace('_', ' ').lower()}"
        availability = "Pre-match (prior completed seasons only)"
        grouping = "Division/season"
    elif name.startswith("Home_Elo_") or name.startswith("Away_Elo_"):
        venue = "Home" if name.startswith("Home_Elo_") else "Away"
        base = name[len(venue)+1:]
        meaning = f"{venue} team's Elo {base.replace('_', ' ').lower()}"
    elif name in ("GA_Cum_Gap", "GF_Cum_Gap", "PPG_Gap", "Rest_Days_Diff"):
        pass  # handled above
    else:
        meaning = name.replace('_', ' ').lower()

    return FeatureSpec(
        name=name,
        dtype=dtype,
        meaning=meaning,
        availability=availability,
        grouping=grouping,
        missing_meaning=missing_meaning,
        fallback=fallback,
        units=units,
        requirement=requirement,
    )


# The canonical schema derived from the current fitted model bundle (84 features).
# This should be kept in sync with the actual model artifact.
def load_canonical_schema() -> FeatureSchema:
    """Load the canonical feature schema from the persisted artifact."""
    # Default path relative to repo root
    repo_root = Path(__file__).parent.parent.parent
    schema_path = repo_root / "output" / "predictions" / "metadata" / "feature_dictionary.json"
    if schema_path.exists():
        data = json.loads(schema_path.read_text())
        # Handle legacy format without version
        if "version" in data:
            return FeatureSchema.from_json(schema_path)
        else:
            # Legacy format: just a list of feature names
            features = data.get("features", [])
            return FeatureSchema.from_fitted_bundle(features)
    # Fallback: construct from BASE_FEATURES in models.py
    from prediction.models import BASE_FEATURES
    return FeatureSchema.from_fitted_bundle(BASE_FEATURES)


if __name__ == "__main__":
    schema = load_canonical_schema()
    print(f"Schema version: {schema.version}")
    print(f"Total features: {len(schema.features)}")
    print(f"Required: {len(schema.required_names())}")
    print(f"Optional: {len(schema.optional_names())}")
    print(f"Class order: {schema.class_order}")
    print("\nFirst 10 features:")
    for f in schema.features[:10]:
        print(f"  {f.name}: {f.meaning} [{f.dtype}]")