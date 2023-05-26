import torch
import torch.nn as nn
from tqdm.auto import tqdm
from timeit import default_timer as timer
from utils import *

#train step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn_binary: torch.nn.Module,
               loss_fn_multiclass: torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device= torch.device):
    """
    Training step function.
    """

    model.train()

    train_loss, train_acc = 0,0

    for batch, (X, y) in enumerate(dataloader):
        y = y.type(torch.LongTensor)
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()

        y_pred_binary1, y_pred_binary2, y_pred_multiclass = model(X)

        loss_binary1 = loss_fn_binary(torch.sigmoid(y_pred_binary1), y[:, 0].unsqueeze(1).float())
        loss_binary2 = loss_fn_binary(torch.sigmoid(y_pred_binary2), y[:, 1].unsqueeze(1).float())
        loss_multiclass = loss_fn_multiclass(y_pred_multiclass, y[:, 2])

        loss = loss_binary1 + loss_binary2 + loss_multiclass

        loss.backward()

        train_loss += loss.item()

        optimizer.step()

        y_pred_binary1_class = torch.round(torch.sigmoid(y_pred_binary1))
        y_pred_binary2_class = torch.round(torch.sigmoid(y_pred_binary2))
        y_pred_multiclass_class = torch.argmax(y_pred_multiclass, dim=1)

        acc_binary1 = ((y_pred_binary1_class == y[:, 0].unsqueeze(1)).sum(dim=1) == 1).sum().item() / len(y_pred_binary1)
        acc_binary2 = ((y_pred_binary2_class == y[:, 1].unsqueeze(1)).sum(dim=1) == 1).sum().item() / len(y_pred_binary2)

        # Calculate accuracy for multiclass head
        acc_multiclass = (y_pred_multiclass_class == y[:, 2]).sum().item() / len(y_pred_multiclass)


        train_acc += (acc_binary1 + acc_binary2 + acc_multiclass) / 3

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def validation_step(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    loss_fn_binary: torch.nn.Module,
                    loss_fn_multiclass: torch.nn.Module,
                    device=torch.device):
    """
    Validation step function.
    """
    # Put model in eval mode
    model.eval()

    validation_loss, validation_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            y = y.type(torch.LongTensor)
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred_binary1, y_pred_binary2, y_pred_multiclass = model(X)

            # Calculate loss for each head
            loss_binary1 = loss_fn_binary(torch.sigmoid(y_pred_binary1), y[:, 0].unsqueeze(1).float())
            loss_binary2 = loss_fn_binary(torch.sigmoid(y_pred_binary2), y[:, 1].unsqueeze(1).float())
            loss_multiclass = loss_fn_multiclass(y_pred_multiclass, y[:, 2])

            validation_loss += (loss_binary1.item() + loss_binary2.item() + loss_multiclass.item()) / 3

            # Calculate accuracy for binary heads
            pred_binary1 = torch.round(torch.sigmoid(y_pred_binary1))
            pred_binary2 = torch.round(torch.sigmoid(y_pred_binary2))

            acc_binary1 = ((pred_binary1 == y[:, 0].unsqueeze(1)).sum(dim=1) == 1).sum().item() / len(pred_binary1)
            acc_binary2 = ((pred_binary2 == y[:, 1].unsqueeze(1)).sum(dim=1) == 1).sum().item() / len(pred_binary2)


            # Calculate accuracy for multiclass head
            pred_multiclass = torch.argmax(y_pred_multiclass, dim=1)
            acc_multiclass = (pred_multiclass == y[:, 2]).sum().item() / len(pred_multiclass)

            validation_acc += (acc_binary1 + acc_binary2 + acc_multiclass) / 3

    validation_loss = validation_loss / len(dataloader)
    validation_acc = validation_acc / len(dataloader)

    return validation_loss, validation_acc



# Create a train function that takes in various model parameters + optimizer + dataloaders + loss function
def train(model: torch.nn.Module,
          train_dataloader,
          validation_dataloader,
          optimizer,
          loss_fn_binary: torch.nn.Module,
          loss_fn_multiclass: torch.nn.Module,
          epochs: int = 5,
          name_save: str = "model",
          device= torch.device):


  """
  Args:
  model : Model to train.
  train_dataloades : Train data iterable.
  validation_dataloader : Validation data iterable.
  optimizer : Optimizer function.
  loss_fn : Loss Function.
  epochs : Number of epochs to do. (default 5)
  device : Device to train the model

  Returns:
  Returns numpy dictionary with train and validation loss and accuracy and training time
  """
  
  results = {"train_loss": [],
             "train_acc": [],
             "validation_loss": [],
             "validation_acc": []}
  
  best_accuracy = 0.0
  models_path = Path().cwd()

  start = timer()
  
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn_binary=loss_fn_binary,
                                       loss_fn_multiclass = loss_fn_multiclass,
                                       optimizer=optimizer,
                                       device=device)
                                       
    validation_loss, validation_acc = validation_step(model=model,
                                                      dataloader=validation_dataloader,
                                                      loss_fn_binary=loss_fn_binary,
                                                      loss_fn_multiclass = loss_fn_multiclass,
                                                      device=device)
    
    if(validation_acc > best_accuracy):
       path = Path( str(name_save) + "_" + str(epoch+1) + "_" + str(epochs) + "_epcs.pth")
       save_model(models_path, path, model)
       best_accuracy = validation_acc
       
    
    # Print out what's happening
    print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"validation_loss: {validation_loss:.4f} | "
          f"validation_acc: {validation_acc:.4f}"
        )

    #Update results 
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["validation_loss"].append(validation_loss)
    results["validation_acc"].append(validation_acc)
  
  end = timer()

  time = end - start
  #Return the  results at the end of the epochs
  return results, time