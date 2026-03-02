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

def test_corrupted_registry_file_raises(tmp_path):
    registry_path = tmp_path / "registry.json"

    # Wrie invalid JSON
    registry_path.write_text("{invalid json")

    with pytest.raises(RuntimeError, match="corrupted"):
        ModelRegistry(registry_path)

def test_invalid_schema_raises(tmp_path):
    registry_path = tmp_path / "registry.json"
    
    registry_path.write_text('{"production": null}') # Missing keys

    with pytest.raises(RuntimeError, match="schema"):
        ModelRegistry(registry_path)

def test_archive_must_be_a_list(tmp_path):
    registry_path = tmp_path / "registry.json"

    bad_data = {
        "production": None,
        "staging": None,
        "archived": "not a list"
    }

    registry_path.write_text(json.dumps(bad_data))

    with pytest.raises(RuntimeError, match="archived"):
        ModelRegistry(registry_path)
                      
def test_deduplicate_and_remove_production_from_archive(tmp_path):
    registry_path = tmp_path / "register.json"

    data = {
        "production": "v1",
        "staging": None,
        "archived": ["v1", "v1", "v2"]
    }
    
    registry_path.write_text(json.dumps(data))
                             
    registry = ModelRegistry(registry_path)
    data = registry._load()

    assert data["archived"] == ["v2"]

def test_version_exists_returns_false(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)

    assert registry.version_exists("non existant") is False

def test_rollback_without_production_raises(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)

    with pytest.raises(RuntimeError, match="No production"):
        registry.rollback_production()

def test_rollback_missing_archived_artifact(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)

    # create valid artifact v1
    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    (v1_dir / "model.joblib").write_text("dummy")

    registry.promote_to_production("v1")

    # manually inject fake archived version
    data = registry._load()
    data["archived"].append("v_missing")
    registry._atomic_save(data)

    with pytest.raises(RuntimeError, match="does not exist on disk"):
        registry.rollback_production()

def test_successful_rollback(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)

    create_fake_model(tmp_path, "v1")
    create_fake_model(tmp_path, "v2")

    registry.promote_to_production("v1")
    registry.promote_to_production("v2")

    registry.rollback_production()

    data = load_registry(registry_path)

    assert data["production"] == "v1"
    assert data["archived"] == ["v2"]

def test_promote_to_staging(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)

    create_fake_model(tmp_path, "v1")

    registry.promote_to_staging("v1")

    data = load_registry(registry_path)

    assert data["staging"] == "v1"

def test_getters(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)

    assert registry.get_production() is None
    assert registry.get_staging() is None