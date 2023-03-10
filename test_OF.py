import numpy as np
import cv2
import os
from tqdm import tqdm


# define a function to calculate the feature/corner optical flow
def compute_corner_optical_flow(prev_image: np.ndarray, current_image: np.ndarray):
    """
    define a function to calculate the feature/corner optical flow
    Args:
        prev_image (np.ndarray): _description_
        current_image (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    prev_image_gray = cv2.cvtColor(prev_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    current_image_gray = cv2.cvtColor(current_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev_image)

    p0 = cv2.goodFeaturesToTrack(prev_image_gray, mask=None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_image_gray, current_image_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(current_image, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    return img


# define a function to calculate the dense optical flow
def compute_dense_optical_flow(prev_image: np.ndarray, current_image: np.ndarray):
    """
    define a function to calculate the dense optical flow
    Args:
        prev_image (np.ndarray): _description_
        current_image (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    old_shape = current_image.shape
    prev_image_gray = cv2.cvtColor(prev_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    current_image_gray = cv2.cvtColor(current_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    assert current_image.shape == old_shape
    hsv = np.zeros_like(prev_image)
    hsv[..., 1] = 255
    flow = None
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_image_gray,
        next=current_image_gray,
        flow=flow,
        pyr_scale=0.8,
        levels=15,
        winsize=5,
        iterations=10,
        poly_n=5,
        poly_sigma=0,
        flags=10,
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


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


# The main function
if __name__ == "__main__":
    # ! Define some GLOBAL parameters
    INITIAL_DIR = "/home/ke/data/tartanair-release1/abandonedfactory/abandonedfactory/Easy/P006"
    # INITIAL_DIR = "/home/ke/data/tartanair-release1/hospital/hospital/Easy/P000"

    Fx = 320.0  # focal length x
    Fy = 320.0  # focal length y
    Cx = 320.0  # optical center x
    Cy = 240.0  # optical center y

    FOV = 90  # field of view /deg

    WIDTH = 640
    HEIGHT = 480

    CAMERA_PARAS = np.array([Fx, Fy, Cx, Cy])

    CAMERA_MAT = np.array([[Fx, 0, Cx], [0, Fy, Cy], [0, 0, 1]])

    CAMERA_MAT_INV = np.linalg.inv(CAMERA_MAT)

    # ! Which one we want to check
    n = 2
    n_t = 1

    # ! Load the data from the tartanair dataset
    imgs_l, depths_l, poses_l = tartan_data_loader(INITIAL_DIR, "left", 0, 20)  # the 300 has a good depth image
    # <nums, 7> rotaiion vector in quaternion form, and we keep it during the whole process

    for i in range(len(imgs_l) - 1):
        test_of = compute_dense_optical_flow(
            imgs_l[i], imgs_l[i + 1]
        )  # actually the performance of the dense is too bad
        cv2.imshow("Desne Optical Flow", test_of.astype(np.uint8))
        keyValue = cv2.waitKey(300)

        if keyValue & 0xFF == ord(" "):  # pause
            cv2.waitKey(0)
        elif keyValue & 0xFF == 27:  # quit
            break

    for i in range(len(imgs_l) - 1):
        test_of = compute_corner_optical_flow(
            imgs_l[i], imgs_l[i + 1]
        )  # actually the performance of the dense is too bad
        cv2.imshow("Corner Optical Flow", test_of.astype(np.uint8))
        keyValue = cv2.waitKey(300)

        if keyValue & 0xFF == ord(" "):  # pause
            cv2.waitKey(0)
        elif keyValue & 0xFF == 27:  # quit
            break
