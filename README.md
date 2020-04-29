# SPOC-Lite

SPOC-Lite (Soil Property and Object Classification Lite) is a light-weight terrain classifier developed in FY16/17 Topical R&TD Next-Gen AutoNav.


## Prerequisite

- Ubuntu 14.04
- ROS Indigo
- catkin_tools (aka `catkin build`)


## Installation

### Get the code

Checkout this repository and [claraty_comm](http://claratyhub.jpl.nasa.gov/claraty-ros/unrestricted/claraty_comm.git) to your ROS workspace.

If you use `wstool`, this can be automated as
```
mkdir ~/spoc_ws && cd ~/spoc_ws
wstool init src http://claratyhub.jpl.nasa.gov/maars/spoc_lite/raw/master/spoc_lite.rosinstall
```
You should see `spoc_lite` and `claraty_msgs` under `src` directory.

### Build
```
catkin build
```

## Usage

### Launch terrain classifier pipeline

```
roslaunch spoc_lite launch.launch camera:=my_camera image:=image_rect
```

### Run demo program

```
roslaunch spoc_lite demo_classifier.launch
roslaunch spoc_lite demo_mapping.launch
```

### Create a terrain classification model from training data

TBD

### Camara parameter settings

The model used in the demonstrations was made from images taken at 2017 March, which was very sunny day. Depending on the weather condition you run spoc_lite, you may need to tune one camera parameter "exposure". Below is how-to. 
1. After you launch "spoc_lite", launch rqt. 
2. Go to configuration -> "XXX"
3. Set the camera parameters as below. 
    - auto_brightness: manual(3)
        - brightness: 0.0 (this is default value)
    - auto_focus" None(5)
        - focus: 0 (this is default value)
    - auto_gain: manual(3)
        - gain: 178 (this is default value)
    - auto_gamma: Off(0)
        - gamma: 10.0 (this is default value)
    - auto_hue: off(0)
        - hue: 2048 (this is default value)
    - auto_iris:None(5)
        - iris:8.0 (this is default value)
    - autop_pan:Off(0)
        - pan:4 (this is default value)
    - auto_saturation: Manual(3)
        - saturation: 1278 (this is default value)
    - auto_sharpness: Manual(3)
        - sharpness: 1532 (this is default value)
    - auto_shutter: Auto(2)
    - auto_white_balance: Manual(3)
        - white_balance_B: 794
        - white_balance_R: 487
4. Finally set the parameter "exposure"
    - auto_exposure: manual(3)
        - exposure: 300-600 (depending on the weather condition, tune this parameter)


## Package documentation

### sand_classifier

Main terrain classifier node.

[ROS API](sand_classifier/README.md)

### ortho_projector

Orthogonal image projection onto local groud plane.

[ROS API](ortho_projector/README.md)


## Contact

If you have any questions, contact the authors:
- Yumi Iwashita (iwashita@jpl.nasa.gov)
- Kyohei Otsu (otsu@jpl.nasa.gov)
