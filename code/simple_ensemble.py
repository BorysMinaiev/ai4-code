import torch
import torch.nn as nn
import os
from pathlib import Path


class SimpleEnsembleModel(nn.Module):
    def __init__(self, state, state_dict=None):
        super(SimpleEnsembleModel, self).__init__()
        self.top = nn.Linear(6, 2)
        self.softmax = nn.Softmax(dim=1)
        self.name = ""
        if state_dict is not None:
            self.name = state_dict
            self.load_state_dict(torch.load(
                state_dict, map_location=state.device))
        self.to(state.device)

    def forward(self, features):
        per_model = self.top(features)
        return self.softmax(per_model)

    def save(self, suffix):
        output_dir = Path(".")
        output_path = os.path.join(
            output_dir, 'simple-ensemble-model-{}.bin'.format(suffix))
        torch.save(self.state_dict(), output_path)
        print("Saved model to {}".format(output_path))
