import torch
import torch.nn as nn
from tqdm.auto import tqdm

#train step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device= torch.device):
    """
    Training step function.
    """

    model.train()

    train_loss, train_acc = 0,0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader) 
    return train_loss, train_acc


#validation step
def validation_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device=torch.device):

  """
  Test step function.
  """
  # Put model in eval mode
  model.eval()

  validation_loss, validation_acc = 0,  0

  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader): 
      X, y = X.to(device), y.to(device)

      validation_pred_logits = model(X)

      loss = loss_fn(validation_pred_logits, y)
      validation_loss += loss.item()

      validation_pred_labels = validation_pred_logits.argmax(dim=1)
      validation_acc += ((validation_pred_labels == y).sum().item()/len(validation_pred_labels))

  validation_loss = validation_loss / len(dataloader)
  validation_acc = validation_acc / len(dataloader)

  return validation_loss, validation_acc


# Create a train function that takes in various model parameters + optimizer + dataloaders + loss function
def train(model: torch.nn.Module,
          train_dataloader,
          validation_dataloader,
          optimizer,
          loss_fn: torch.nn.Module,
          epochs: int = 5, 
          device= torch.device):


  """
  Args:
  model -> Model to train.
  train_dataloades -> Train data iterable.
  validation_dataloader -> Validation data iterable.
  optimizer -> Optimizer function.
  loss_fn -> Loss Function.
  epochs -> Number of epochs to do. (default 5)
  device -> Device to train the model

  Returns:
  Returns numpy dictionary with train and validation loss and accuracy.
  
  """
  
  results = {"train_loss": [],
             "train_acc": [],
             "validation_loss": [],
             "validation_acc": []}
  
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)
                                       
    validation_loss, validation_acc = validation_step(model=model,
                                                      dataloader=validation_dataloader,
                                                      loss_fn=loss_fn,
                                                      device=device)
    
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
  
  #Return the  results at the end of the epochs
  return results