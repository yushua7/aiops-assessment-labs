import os
import pathlib

# Must run before app.main is imported: the model is loaded at import time.
os.environ.setdefault("MODEL_DIR", str(pathlib.Path(__file__).resolve().parents[1] / "model"))
