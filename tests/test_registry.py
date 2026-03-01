import json
from pathlib import Path

import pytest

from registry import ModelRegistry

def create_fake_model(base_path: Path, version: str):
    version_dir = base_path / version
    version_dir.mkdir(parents=True)
    (version_dir / "model.joblib").write_text("fake_model")

def load_registry(path: Path):
    with open(path) as f:
        return json.load(f)
    
def test_initalisation(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)

    data = load_registry(registry_path)

    assert data["production"] is None
    assert data["staging"] is None
    assert data["archived"] == []

def test_promote_to_production_pushes_archive(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)

    create_fake_model(tmp_path, "v1")
    create_fake_model(tmp_path, "v2")

    registry.promote_to_production("v1")
    registry.promote_to_production("v2")

    data = load_registry(registry_path)

    assert data["production"] == "v2"
    assert data["archived"] == ["v1"]

def test_rollback_fails_if_no_archive(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)

    create_fake_model(tmp_path, "v1")
    registry.promote_to_production("v1")

    with pytest.raises(RuntimeError):
        registry.rollback_production()

def test_invariant_production_not_in_archive(tmp_path):
    registry_path = tmp_path / "registry.json"
    
    # Manually write corrupted registry
    corrupted_data = {
        "production": "v1",
        "staging": None,
        "archived": ["v1", "v1"]
    }

    registry_path.write_text(json.dumps(corrupted_data))

    create_fake_model(tmp_path, "v1")

    registry = ModelRegistry(registry_path)
    
    data = load_registry(registry_path)

    assert data["archived"] == []
