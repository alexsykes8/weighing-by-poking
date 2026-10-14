import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; the default GLFW backend needs a display
sys.path.insert(0, str(Path(__file__).resolve().parent))
