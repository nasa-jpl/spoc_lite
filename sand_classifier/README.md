sand_classifier
===============

Terrain classifier for sand by Yumi Iwashita <iwashita@jpl.nasa.gov>.

## ROS API

### 1. sand_classifier_node

#### 1.1 Published topics

`image` (sensor_msgs/Image)
- Input color image

#### 1.2 Subscribed topics

`image_overlay` (sensor_msgs/Image)
- Overlayed image

`image_label` (sensor_msgs/CameraInfo)
- Image containing terrain labels. 255=sand, 0=otherwise

`image_probability` (sensor_msgs/CameraInfo)
- Image containing sand probability


#### 1.3 Parameters

`~threshold` (double, default: 0.8)
- Threshold for sand probability. Pixels with higher probability are regarded as sand.

`~model_name` (str)
- Path to the training model name.
- Pretrained models are found under `models/`

`~setting_file` (str)
- Setting file for classifier.
- See example files under `spoc_lite/`.

`~use_*` (bool)
- Flags to enable each feature extraction.


### 2. sand_classifier_nodelet

Identical to the `sand_classifier_nodelet`.
