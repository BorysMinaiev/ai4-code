from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path
    unixcoder_model_path: str
    wandb_key: str
    batch_size: int


def get_local_config():
    return Config(data_dir=Path('/home/borys/AI4Code/input/AI4Code'), unixcoder_model_path='/home/borys/AI4Code/input/unixcoderbase', wandb_key='/home/borys/wandb_key', batch_size=2)


def get_jarvis_config():
    return Config(data_dir=Path('/home/input/AI4Code'), unixcoder_model_path='/home/unixcoderbase', wandb_key='/home/wandb_key', batch_size=60)
