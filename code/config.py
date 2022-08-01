from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Config:
    data_dir: Path
    unixcoder_model_path: str
    wandb_key: str
    batch_size: int
    batch_size_graph2: int
    cosine_minibatch_size: int
    cosine_batch_size: int
    use_simple_ensemble_model = True


def get_local_config():
    return Config(data_dir=Path('/home/borys/AI4Code/input/AI4Code'),
                  unixcoder_model_path='/home/borys/AI4Code/input/unixcoderbase',
                  wandb_key='/home/borys/wandb_key',
                  batch_size=2,
                  batch_size_graph2=2,
                  cosine_minibatch_size=2,
                  cosine_batch_size=4
                  )


def get_jarvis_config():
    return Config(data_dir=Path('/home/input/AI4Code'),
                  unixcoder_model_path='/home/unixcoderbase',
                  wandb_key='/home/wandb_key',
                  batch_size=50,
                  batch_size_graph2=30,
                  cosine_minibatch_size=8,
                  cosine_batch_size=60
                  )


def get_default_config():
    if os.getenv('LOGNAME') == 'borys':
        print('Get local config')
        return get_local_config()
    else:
        print('Get jarvis config')
        return get_jarvis_config()
