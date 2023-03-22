from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from PIL import Image
import os
from pathlib import Path
from utils import *
from allResNets import *
from engine import *
from data_transforms import create_transform

NUM_WORKERS = os.cpu_count()

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory."""

    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names could not be found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}... please check file structure.")

    # 3. Create a dictionary of index labels (computers prefer numbers rather than strings as labels)

    class_to_idx = {}
    for i, class_name in enumerate(classes):
        id = np.zeros(len(classes))
        id[i] = 1
        class_to_idx[class_name] = id
        
    return classes, class_to_idx

class CustomImageFolderMultiLabel(torch.utils.data.Dataset):
    def __init__(self, root, transform = None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes, self.class_to_idx = find_classes(root)
        for i, class_name in enumerate(self.classes):
            class_path = os.path.join(root, class_name)
            for image_path in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, image_path))
                label = [0]*len(self.classes)
                label[i] = 1
                self.labels.append(torch.tensor(label))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.image_paths)
    


NUM_WORKERS = os.cpu_count()
def create_dataloaders_multilabel(
    data_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers:int = NUM_WORKERS):

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
    train_data = CustomImageFolderMultiLabel(root=train_dir,
                                    transform=transform) # a transform for the data
    
    validation_data = CustomImageFolderMultiLabel(root=validation_dir,
                                    transform=transform)

    test_data = CustomImageFolderMultiLabel(root=test_dir,
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