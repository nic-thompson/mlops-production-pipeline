from pathlib import Path
import joblib

from src.registry import ModelRegistry

ARTIFACT_ROOT = Path("artifacts/models")
REGISTRY_PATH = ARTIFACT_ROOT / "registry.json"

class ModelLoader:

    def __init__(self):
        self.registry = ModelRegistry(REGISTRY_PATH)
    
    def load_model(self, version: str):
        model_path = ARTIFACT_ROOT / version / "model.joblib"

        if not model_path.exists():
            raise RuntimeError(f"Model artifact not found {model_path}")
        
        return joblib.load(model_path)
    
    def load_production_model(self):
        version = self.registry.get_production()

        if version is None:
            raise RuntimeError("No production model is registered.")
    
        return self.load_model(version)    

    
    def load_staging_model(self):
        version = self.registry.get_staging()

        if version is None:
            raise RuntimeError("No staging model is registered.")

        return self.load_model(version)    
    