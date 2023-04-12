# ToolKit
This is a repository to store some tool kits created by Ke for the future use. 
### 1. _robot_transformation.py_ is mainly for some transformation between different frame or from pose vector to the pose matrix
### 2. _pointcloud_kit.py_ is mainly for the create of the pointcloud and the visualization of the pointcloud, to use this module, the _robot_transformation.py_ module is necessary.
### 3. _test_OF.py_ is a small demo to create the Optical Flow based on the opencv and some skills about how to show the image and use the keyboard to control the window, during this demo I use the [Tartanair dataset](https://theairlab.org/tartanair-dataset/) as the example, there is a customed tartanair data loader, if you wanna try this demo, first you should change the `INITIAL_DIR`.
### 4. _evaluation_tools.py_ is a evaluatioin/metrics tool of SLAM/VO performance, it contains the common metrics ATE, RPE, KITTI score and one customized metric called the Success Rate to check the performance of the algorithm on one trajectory, to use this module, the _robot_transformation.py_ module is necessary.
### 5. _icp.py_ is a icp module for the 2D icp
### 6. _draw_tools.py_ is a module to draw the trajectory for the camera pose, the result should be like this: 
<p align="center">
<img src="https://github.com/GUOkekkk/ToolKit/blob/main/draw_test.png" alt= “test_draw” width="600" />
</p>
### 7._dataloader.py_ is a module to load the data and now it only supports the tartanair_data
