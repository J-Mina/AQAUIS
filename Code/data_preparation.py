import os
import cv2
from pathlib import Path
import glob
import torch
import random
from shutil import copyfile


def convert_mp4_to_png(video_path, frame_path):
  """
    Convert mp4 videos to png images.
    
    Args:
    video_path : Path to the video to convert with the video name.
    frame path : Path to the folder in which frames should be saved and the video name as index to add the frame info.
  """
  vidcap = cv2.VideoCapture(video_path)
  success,image = vidcap.read()
  count = 0
  while success:
    frame_path_mod = frame_path+"_frame%d.png" % count
    cv2.imwrite(frame_path_mod, image)     # save frame as JPEG file      
    success,image = vidcap.read()

    if(count % 500 == 0):
      print(f"Frame:{count} | Success:{success}")
    count += 1

def check_dir(DATASET_PATH):
    """
        Verifies if the path of the Dataset is created, if not creates it.
    """
    if DATASET_PATH.is_dir():
        pass
    else:
        print(f"{DATASET_PATH} does not exist, creating one...")
        DATASET_PATH.mkdir(parents=True, exist_ok=True)


def walk_through_dir(dir_path):
  """Walks through dir_path returning its contents."""
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} frames in '{dirpath}'.")


def convert_all(DATASET_PATH, LIST_PATHS, RAW_DATA_PATH):
    """
    Convert all videos into images.

    Args:
    DATASET_PATH : path to where the dataset is split into classes.
    LIST_PATHS : list with the classes folders in which the data is converted to.
    RAW_DATA_PATH : path to the raw data.
    """
    check_dir(DATASET_PATH)

    for i in range(len(LIST_PATHS)):
        folders = os.path.split(LIST_PATHS[i])[0]
        dataset_full_path_folders = DATASET_PATH + folders
        check_dir(Path(dataset_full_path_folders))

        raw_full_path_folders = RAW_DATA_PATH + LIST_PATHS[i]

        videos = glob.glob(raw_full_path_folders+"/*.mp4")

        for j, video in enumerate(videos):
            video_name = dataset_full_path_folders + "/" + os.path.split(LIST_PATHS[i])[1] + "_" + str(j)
            convert_mp4_to_png(video,video_name)

def split_data(data_source, data_folder, split_size, num_img_class = 0):
    """
    Split all of the data into training, validation and testing with a split size.

    Args:
    data_source : Folder that has the dataset folder.
    data_folder : Folder in which the data should be splitted to.
    split_size : Ratio of training to testing data from [trainning(0-1),validation(0-1),testing(0-1)] make sure that the sum is 1.
    num_img_class : Number of images per class to keep in the dataset. (For the creation of smaller datasets or in this case balancing).
    
    """

    #Change disk directory
    base_path = Path("G:/Dissertation/")
    if(Path().cwd() != Path(r"G:\Dissertation")):
        os.chdir(base_path)

    #Verify if the folder exists, if not creates it
    check_dir(data_folder)

    data_train = data_folder / Path('train/')
    data_validation = data_folder / Path('validation/')
    data_test = data_folder / Path('test/')

    check_dir(data_train)
    check_dir(data_validation)
    check_dir(data_test)

    if(sum(split_size) != 1):
        print("SPLIT_SIZE is not valid")
        return
    
    check_dir(Path(data_train))
    check_dir(Path(data_test))
    check_dir(Path(data_validation))

    for dir in os.listdir(data_source):
            training = data_train / dir
            testing = data_test / dir
            validation = data_validation / dir
            check_dir(Path(training))
            check_dir(Path(testing))
            check_dir(Path(validation))

            source = data_source / dir

            files = []
            print('Split Data')
            for filename in os.listdir(source):
                file = source / filename
                if os.path.getsize(file) > 0:
                    files.append(filename)
                else:
                    print(filename + "is zero lenght, so ignoring.")
          
            num_imgs = len(files)

            if(num_img_class != 0 ):
              num_imgs = num_img_class
          
            training_length = int(num_imgs* split_size[0])
            validation_length = int(num_imgs* split_size[1])
            testing_length = int(num_imgs* split_size[2])

            print('SOURCE: ',source, '\n TRAINING', training, '\n ',num_imgs)
            print('training_length:',training_length)
            print('validation_length:',validation_length)
            print('testing_length:',testing_length)

            shuffled_set = random.sample(files, num_imgs)
            training_set = shuffled_set[0:training_length]
            validation_set = shuffled_set[training_length:training_length+validation_length]
            testing_set= shuffled_set[training_length+validation_length:]
            
            for filename in training_set:
                this_file = source / filename
                destination = training / filename
                copyfile(this_file, destination)

            for filename in validation_set:
                this_file = source / filename
                destination =validation / filename
                copyfile(this_file, destination)
            
            for filename in testing_set:
                this_file = source / filename
                destination = testing / filename
                copyfile(this_file, destination)
        



