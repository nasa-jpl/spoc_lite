ortho_projector
===============

This package projects camera images onto ground, assuming the local terrain is flat.

## ROS API

### 1. ortho_projector_node

Project images onto ground.

#### 1.1 Published topics

`map` (claraty_msgs/FloatGrid)
- 2D grid map containing a float value in each cell

`map/image_raw` (sensor_msgs/Image)
- Orthoprojected image in original image color format

`map/image_color` (sensor_msgs/Image)
- Orthoprojected image in RGB color

#### 1.2 Subscribed topics

`image` (sensor_msgs/Image)
- Input image

`camera_info` (sensor_msgs/CameraInfo)
- Camera calibration info

`reset` (std_msgs/Empty)
- Clear map
- TODO: This will be reimplemented as a service.

#### 1.3 Services

None

#### 1.4 Parameters

`~length` (double, default: 50.0)
- The length (x) dimensions of the map [m].

`~width` (double, default: 50.0)
- The width (y) dimensions of the map [m].

`~cell_size` (double, default: 0.25)
- The edge length of square cells in the map [m].

`~map_frame_id` (str, default: "MAP")
- Frame ID of the map.

`~base_frame_id` (str, default: "RNAV")
- Frame ID of the rover base.

`~camera_frame_id` (str, default: "NCAML")
- Frame ID of the sensor.

`~merge_mode` (str, default: "LOT")
- Merge mode. Candidates are First on Top (FOT), Last on Top (LOT), Maximum (MAX), Minimum (MIN).

`~margin_top`, `~margin_bottom`, `~margin_left`, `~margin_right` (int, default: 0)
- Boundary image region to be extracted from processing [pixel].


### 2. ortho_projector_nodelet

Identical to the `ortho_projector_node`.


### 3. terrain_mapper_node

Publish terrain class map containing hazard probability (e.g., sand).

#### 3.1 Published topics

`map` (claraty_msgs/TerrainClassMap)
- 2D grid map containing terrain classification result.

`map_int` (nav_msgs/OccupancyGrid)
- 2D grid map containing hazard probability

#### 3.2 Subscribed topics

`map_float` (claraty_msgs/FloatGrid)
- Orthoprojected hazard probability

#### 3.3 Parameters

`~threshold` (float, default: 0.4)
- Threshold of hazard probability. A cell with higher probability is regarded as hazard.


### 4. terrain_mapper_nodelet

Identical to the `terrain_mapper_node`.
