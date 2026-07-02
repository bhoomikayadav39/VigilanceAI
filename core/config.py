from pathlib import Path
from pydantic_settings import BaseSettings

class EngineSettings(BaseSettings):
    """Enforces strict, type-safe environment parameters across the pipeline."""
    PROJECT_NAME: str = "VigilanceAI"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: Path = Path(__file__).resolve().parent.parent / "storage"
    
    # Mathematical thresholds for frame matrix deviation
    PIXEL_VARIANCE_THRESHOLD: float = 35.0
    MIN_ANOMALY_FRAMES_TRIGGER: int = 5
    
    # Model Execution Params
    VISION_MODEL_ID: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    USE_CUDA_IF_AVAILABLE: bool = True

    class Config:
        env_file = ".env"

settings = EngineSettings()

if __name__ == "__main__":
    print(f"[✓] {settings.PROJECT_NAME} Config Pipeline verified.")
    print(f"[*] Native Storage Endpoint target: {settings.STORAGE_DIR}")