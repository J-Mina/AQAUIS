from pathlib import Path
import torch
from data_preparation import check_dir
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_model(model_path, model_name, model):

    check_dir(model_path)

    model_save_path = model_path / model_name

    print(f"Saving the model to: {model_save_path}")
    torch.save(model.state_dict(),
               model_save_path)



def load_model(model, model_path):
    """
    Not working.
    """

    loaded_model = model()
    loaded_model.load_state_dict(torch.load(model_path))

    return loaded_model



def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, 
               accuracy_fn,
               device : torch.device):
  """Returns a dictionary containing the results of model predicting on data_loader."""
  loss, acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in tqdm(data_loader):

      X = X.to(device)
      y = y.to(device)

      # Make predictions
      y_pred = model(X)

      # Accumulate the loss and acc values per batch
      loss += loss_fn(y_pred, y)
      acc += accuracy_fn(y_true=y,
                         y_pred=y_pred.argmax(dim=1))

    # Scale loss and acc to find the average loss/acc per batch
    loss /= len(data_loader)
    acc /= len(data_loader)

  return {"model_name": model.__class__.__name__, # only works when model was created with a class
          "model_loss": loss.item(),
          "model_acc": acc}


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.
    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.
    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    validation_loss = results['validation_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    validation_accuracy = results['validation_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['validation_loss']))

    # Setup a plot 
    plt.figure(figsize=(10, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, validation_loss, label='validation_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, validation_accuracy, label='validation_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def get_predictions(model, dataloader, device):
    model.eval()
    images=[]
    labels=[]
    probs = []

    with torch.inference_mode():
        for(X, y) in dataloader:
            X = X.to(device)
            y_pred  = model(X)
            y_prob = F.softmax(y_pred, dim=-1)
            images.append(X.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs



def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels,pred_labels)
    cm = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)
    cm.plot(values_format='d', cmap='Blues', ax=ax)

def change_to_disk():
    #Change disk directory
    base_path = Path("G:/Dissertation/")
    if(Path().cwd() != Path(r"G:\Dissertation")):
        os.chdir(base_path)