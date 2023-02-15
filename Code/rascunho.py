import os
import cv2
from pathlib import Path
import random
import torch
import glob
from shutil import copyfile


#Change disk directory
base_path = Path("G:/Dissertation/")
if(Path().cwd() != Path(r"G:\Dissertation")):
    os.chdir(base_path)



#Define data_paths
raw_data_path = Path("raw_data/Data examples/")
raw_visibility_path = raw_data_path / Path("1_Visibility/")
raw_quality_path = raw_data_path/ Path("2_Quality/")
dataset_path = Path('dataset/')


from data_preparation import split_data

folder = Path("/small_split")
split_size = [0.8,0.1,0.1]

split_data(dataset_path, folder, split_size, num_img_class=1000)