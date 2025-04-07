# Helper Scripts for Data Recording

These scripts facilitate the data collection process for the dataset of TinyCenterSpeed.

The topics, which should be recorded can be configured in `topics_to_record.txt` on new lines.
The Rosbag name and duration can be easily configured in `record_rosbag.sh`.

To record the Rosbags run: `bash record_rosbag.sh`


The following topics are required for to use the dataset building mechnanism:
```
/opp/car_state/odom
/opp/car_state/odom_frenet
/car_state/odom
/car_state/odom_frenet
/car_state/pose
/lap_data
/scan
/tf
/tf_static
/map
/global_waypoints
```
The topics which are part of the /opp correspond to data coming from state estimation of an adversial car.
During dataset generation they form the ground truth signals.
