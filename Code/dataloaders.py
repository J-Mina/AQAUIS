from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    data_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers:int=NUM_WORKERS):

    """
    Create dataloaders for the data split into train/validation/test

    Args:
    data_dir : Path to data directory with (train/validation/test split).
    transform : torchvision transforms to perform on training, validation and testing data.
    batch_size : Number of samples per batch in each of the DataLoaders.
    num_workers : An integer for number of workers per DataLoader.

    Returns:
    
    train dataloader, validation_dataloader, test_dataloader, train_data, validation_data, test_data, class_names
    """

    train_dir = data_dir / "train/"
    validation_dir = data_dir / "validation/"
    test_dir = data_dir / "test/"


    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(root=train_dir,
                                    transform=transform, # a transform for the data
                                    target_transform=None) # a transform for the label/target
    
    validation_data = datasets.ImageFolder(root=validation_dir,
                                    transform=transform)

    test_data = datasets.ImageFolder(root=test_dir,
                                    transform=transform)


    #Get classes
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False)

    validation_dataloader = DataLoader(dataset=validation_data,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader, train_data, validation_data, test_data, class_names
