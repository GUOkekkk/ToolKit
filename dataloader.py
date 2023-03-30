import numpy as np
import cv2
import os
from tqdm import tqdm

# define a data loader to load the img, depth and camera pose from the tartan dataset
def tartan_data_loader(path: str, position: str, start_num: int = 0, end_num: int = 5):

    """
    Load the tartanair data set

    Args:
        _path (str): the path of the dataset
        _position (str): left or right
        _start_num (int): the start idx of the sequence
        _end_num (int): the end idx of the sequence, if it is over1000 it will be set as the length of the seauence

    Returns:
        _img (np.ndarray): the color img <n, 480, 640, 3>
        _depth (np.ndarray): the depth img <n, 480, 640, 3>
        _camera_pose (np.ndarray): the camera pose <n, 7>
    """
    # Set the path
    _image_dir = path + "/image_" + position
    _depth_dir = path + "/depth_" + position
    _pose_dir = path + "/pose_" + position + ".txt"

    img_sequence = np.zeros((0, 480, 640, 3))  # Create a empty array to store the img

    _image_dir_sequences = os.listdir(_image_dir)
    end_num = len(_image_dir_sequences) if end_num > len(_image_dir_sequences) else end_num

    _image_dir_sequences.sort(key=lambda x: int(x.split("_")[0]))  # Solve the problem that the list is not in order

    # _test_video = cv2.VideoWriter("VideoTest.avi", cv2.VideoWriter_fourcc("I", "4", "2", "0"), 5, (640, 480))
    print("Load the data in", path)
    print("Load the color image:")
    for _img_name in tqdm(_image_dir_sequences[start_num:end_num]):
        _img = cv2.imread(_image_dir + "/" + _img_name)
        # _test_video.write(_img)
        _img = np.expand_dims(_img, axis=0)  # expend the dim for the concatenate
        img_sequence = np.concatenate((img_sequence, _img), axis=0)

    # _test_video.release()
    # cv2.destroyAllWindows()

    depth_sequence = np.zeros((0, 480, 640))  # Create a empty array to store the depth

    _depth_dir_sequences = os.listdir(_depth_dir)
    _depth_dir_sequences.sort(key=lambda x: int(x.split("_")[0]))  # Solve the problem that the list is not in order

    print("Load the depth image:")
    for _depth_name in tqdm(_depth_dir_sequences[start_num:end_num]):
        _depth = np.load(_depth_dir + "/" + _depth_name)
        _depth = np.expand_dims(_depth, axis=0)  # expend the dim for the concatenate
        depth_sequence = np.concatenate((depth_sequence, _depth), axis=0)

    pose_sequence = np.loadtxt(_pose_dir)[start_num:end_num]

    return img_sequence, depth_sequence, pose_sequence
