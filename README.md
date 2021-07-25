# 3D SORT Tracker
This tracker was inspired by the [2D SORT tracker](https://github.com/abewley/sort). Their code was adapted to work using 3D bounding boxes instead of just 2D bounding boxes.

## How to use
Go into the src folder of your catkin workspace and run the following
```
git clone [this repo]
cd sort_3d
pip install -r requirements.txt
```

Then catkin_make or build your catkin workspace.

The tracker can be launched by running
```
roslaunch sort_3d sort_3d.launch
```

Note: This by default also starts the plotter node (can use rviz to visualize the markers), see the launch file for more information.

## Requirements
This node subscribes to a topic of type pcdet_ros_msgs.msg.BoundingBoxes3D, make sure that you have that added to your catkin workspace. The version that is working with this version has been added to the extras folder for future reference.
