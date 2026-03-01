import json
from pathlib import Path
from typing import Optional

class ModelRegistry:
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.registry_path.exists():
            self._initialize_registry()

        self._validate_integrity()

    # --------------------
    # Internal helpers
    # --------------------  

    def _initialize_registry(self):
        data = {
            "production": None,
            "staging": None,
            "archived": []
        }
        self._atomic_save(data)

    def _load(self) -> dict:
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError("Registry file is corrupted.") from e

    def _atomic_save(self, data: dict):
        tmp_path = self.registry_path.with_suffix(".tmp")

        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=4)

        tmp_path.replace(self.registry_path)

    def _validate_integrity(self):
        data = self._load()

        required_keys = {"production", "staging", "archived"}
        if not required_keys.issubset(data.keys()):
            raise RuntimeError("Registry schema invalid.")
        
        if not isinstance(data["archived"], list):
            raise RuntimeError("'archived' must be a list.")
        
        # Remove duplicates defensively
        data["archived"] = list(dict.fromkeys(data["archived"]))

        # Production must not be in archive
        if data["production"] in data["archived"]:
            data["archived"].remove(data["production"])
        
        # Staging must not be in archive
        if data["staging"] in data["archived"]:
            data["archived"].remove(data["staging"])

        self._atomic_save(data)

    def _artifact_exists(self, version: str) -> bool:
        version_path = self.registry_path.parent / version
        model_file = version_path / "model.joblib"
        return version_path.is_dir() and model_file.exists()

    # --------------------
    # Public API
    # --------------------

    def version_exists(self, version: str) -> bool:
        return self._artifact_exists(version)
    
    def get_production(self):
        return self._load()["production"]
    
    def get_staging(self):
        return self._load()["staging"]
    
    def promote_to_staging(self, version: str):
        if not self.version_exists(version):
            raise ValueError(f"Version '{version}' does not exist.")
        
        data = self._load()
        data["staging"] = version
        self._atomic_save(data)
    
    def promote_to_production(self, version: str):
        if not self.version_exists(version):
            raise ValueError(f"Version '{version}' does not exist.")
        
        data = self._load()

        current_prod = data["production"]
        if current_prod and current_prod != version:
            if current_prod not in data["archived"]:
                    data["archived"].append(current_prod)
        
        data["production"] = version
        data["staging"] = None
        
        self._atomic_save(data)

    def rollback_production(self):
        data = self._load()

        if data["production"] is None:
            raise RuntimeError("No production model set. Cannot rollback.")
        
        if not data["archived"]:
            raise RuntimeError("No archived versions available for rollback.")
        
        current_prod = data["production"]
        previous_version = data["archived"][-1] # LIFO semantics

        # Ensure previous version still exists on disk
        if not self._artifact_exists(previous_version):
            raise RuntimeError(
                f"Archived version '{previous_version}' does not exist on disk."
            )
        data["archived"].pop()
        
        # Move current production into archive 
        data["archived"].append(current_prod)

        # Promote previous version to production
        data["production"] = previous_version                      
                
        self._atomic_save(data)