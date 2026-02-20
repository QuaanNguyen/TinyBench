from app.config import MODELS_DIR

def list_models():
    return list(MODELS_DIR.glob("**/*.gguf"))