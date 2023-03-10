from torchvision import transforms
from PIL import Image
import random
import torch


    
def colorDeviation(image: Image.Image) -> Image.Image:
    """
    Insert the usual color deviation of underwater images
    """
    #Add probability
    num = random.random()

    if num > 0.5 : 
        #split image into color channels
        b, g, r = image.split()

        #dimming factor of the green channel
        gnum = random.uniform(0.6,1)

        #dimming factor of the red channel
        rnum = random.uniform(0,0.3)
        
        #Apply the dimming factor
        r = r.point(lambda i: i * rnum)
        g = g.point(lambda i: i * gnum)

        #return the resulting image
        img = Image.merge('RGB', (r,g,b))
        return img
    
    else:
        return image

def create_transform(
        resize : tuple,
        rotate : int = 0,
        color_dev : bool = True,
        transf_tensor : bool = True,
        normalize : bool = True):
    
    if(sum(resize) != 0):
        resize_cmd = transforms.Resize(size=(resize[0],resize[1]))
    else:
        resize_cmd = transforms.RandomHorizontalFlip(p=0)
    
    if(rotate != 0):
        rotate_cmd = transforms.RandomRotation(rotate)
    else:
        rotate_cmd = transforms.RandomHorizontalFlip(p=0)
    
    if(color_dev):
        color_dev_cmd = transforms.Lambda(colorDeviation)
    else:
        color_dev_cmd = transforms.RandomHorizontalFlip(p=0)

    if(transf_tensor):
        transf_cmd = transforms.ToTensor()
    else:
        transf_cmd = transforms.RandomHorizontalFlip(p=0)

    if(normalize):
        normalize_cmd = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    else:
        normalize_cmd = transforms.RandomHorizontalFlip(p=0)

    data_transform = transforms.Compose([
        resize_cmd,
        rotate_cmd,
        color_dev_cmd,
        transf_cmd,
        normalize_cmd
    ])

    return data_transform