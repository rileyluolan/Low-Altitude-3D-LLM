"""One path contract shared by downloads, construction, training and evaluation."""
import os
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
def env_path(name, default):
    value = Path(os.environ.get(name, str(default))).expanduser().resolve()
    os.environ.setdefault(name, str(value))
    return value
os.environ.setdefault("EXPERIMENT_ROOT", str(ROOT))
WORKSPACE = env_path("WORKSPACE_ROOT", ROOT.parents[1])
STARVLA = env_path("STARVLA_ROOT", ROOT / "third_party/starVLA")
HUGEBENCH = env_path("HUGEBENCH_ROOT", ROOT / "third_party/HUGE-Bench")
GAUSSIAN = env_path("GAUSSIAN_SPLATTING_ROOT", ROOT / "third_party/gaussian-splatting")
HUGE_DATA = env_path("HUGE_DATA_ROOT", ROOT / "data/HUGE_data")
MODEL = env_path("MODEL_ROOT", ROOT / "models/Qwen3-VL-2B-RefDrone-Stage2")
VENV = env_path("VENV_ROOT", ROOT / ".venv")
BASE = env_path("BASE_CKPT", ROOT / "artifacts/base/qwen3vl_2b_pi_v3_hugebench_refdrone_base/checkpoints/base_pytorch_model.pt")

def load_config(path):
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return cfg

def config_dict(path):
    return OmegaConf.to_container(load_config(path), resolve=True)
