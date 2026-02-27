import json
from pathlib import Path

class ModelRegistry:
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.registry_path.exists():
            self.initialize_registry()

    def _initialize_registry(self):
        data = {
            "production": None,
            "staging": None,
            "archived": []
        }
        self._save(data)

    def _load(self):
        with open(self.registry_path, "r") as f:
            return json.load(f)
    
    def _save(self, data: dict):
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=4)

    def get_production(self):
        return self._load()["production"]
    
    def get_staging(self):
        return self.load()["staging"]
    
    def promote_to_staging(self, version: str):
        data = self._load()
        data["staging"] = version
        self._save(data)
    
    def promote_to_production(self, version: str):
        data = self._load()

        current_prod = data["production"]
        if current_prod:
            data["archived"].append(current_prod)
        
        data["production"] = version
        data["staging"] = None
        self._save(data)