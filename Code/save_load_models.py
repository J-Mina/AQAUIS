from pathlib import Path
import torch
from data_preparation import check_dir
import os

def save_model(model_path, model_name, model):

    check_dir(model_path)

    model_save_path = model_path / model_name

    print(f"Saving the model to: {model_save_path}")
    torch.save(model.state_dict(),
               model_save_path)



def load_model(model, model_path):

    loaded_model = model()
    loaded_model.load_state_dict(torch.load(model_path))

    return loaded_model
