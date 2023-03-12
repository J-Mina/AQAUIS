from torchvision import transforms
from PIL import Image
import random
import torch
import numpy as np
from skimage.util import random_noise

    
def colorDeviation(image: Image.Image) -> Image.Image:
    """
    Insert the usual color deviation of underwater images with a probability of 50%

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The noisy image.
    """
    #Add probability
    num = random.random()

    if num > 0.5 : 
        #split image into color channels
        b, g, r = image.split()

        #dimming factor of the blue channel
        bnum = random.uniform(0.9,1)

        #dimming factor of the green channel
        gnum = random.uniform(0.6,1)

        #dimming factor of the red channel
        rnum = random.uniform(0,0.3)
        
        #Apply the dimming factor
        b = b.point(lambda i: i * bnum)
        r = r.point(lambda i: i * rnum)
        g = g.point(lambda i: i * gnum)

        #return the resulting image
        img = Image.merge('RGB', (r,g,b))
        return img
    
    else:
        return image
    
def SPNoise(image: Image.Image)->Image.Image:
    
    """
    Add salt and pepper noise to a PIL image.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The noisy image.
    """
     
    amount = 0.01

    # Convert the PIL image to a numpy array
    np_image = np.array(image)

    # Generate a random mask for salt and pepper noise
    mask = np.random.rand(*np_image.shape[:2])

    # Add salt and pepper noise to the image
    np_image[mask < amount / 2] = 0
    np_image[mask > 1 - amount / 2] = 255

    # Convert the numpy array back to a PIL image
    noisy_image = Image.fromarray(np_image)

    return noisy_image

def GaussianNoise(image: Image.Image)->Image.Image:

    mean = 0
    var = 0.01

    # Convert the PIL image to a numpy array
    np_image = np.array(image)

    # Generate Gaussian noise with the given mean and variance
    noise = np.random.normal(mean, var ** 0.5, np_image.shape)

    # Add the noise to the image
    noisy_image = np_image + noise

    # Clip the values to the range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    # Convert the numpy array back to a PIL image
    noisy_image = Image.fromarray(np.uint8(noisy_image))

    return noisy_image


def create_transform(
        resize : tuple,
        rotate : int = 0,
        color_dev : bool = True,
        transf_tensor : bool = True,
        normalize : bool = True,
        sp_noise : bool = False,
        gauss_noise : bool = False):
    
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

    if(sp_noise):
        sp_noise_cmd = transforms.Lambda(SPNoise)
    else:
        sp_noise_cmd = transforms.RandomHorizontalFlip(p=0)

    if(gauss_noise):
        gauss_noise_cmd = transforms.Lambda(GaussianNoise)
    else:
        gauss_noise_cmd = transforms.RandomHorizontalFlip(p=0)


    data_transform = transforms.Compose([
        resize_cmd,
        rotate_cmd,
        color_dev_cmd,
        sp_noise_cmd,
        gauss_noise_cmd,
        transf_cmd,
        normalize_cmd
        
    ])

    return data_transform