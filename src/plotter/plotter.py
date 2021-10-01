#!/usr/bin/env python3

import rospy

import numpy as np

from typing import Union, List
from pyquaternion import Quaternion

from visualization_msgs.msg import MarkerArray, Marker
from arl_msgs.msg import BBox3D, BBox3DArray

def numpy_to_MarkerMsg(box_arr: Union[np.array, List[float]], name_space:str, idx:int, frame_id:str='') -> Marker:
    '''
    Construct a marker to plot a bounding box given its location, scale, heading, among other parameters
    
    Parameters
    ----------
    box_arr: np.array (or list) of floats, shape=(7, )
        Contains [x, y, z, dx, dy, dz, heading] of the box
    name_space: str
        Namespace to add the marker to
    idx: int
        Unique id of the box, used to assign a unique color to the box
        Note: for markers, the name_space and idx combined should be unique for each box
    frame_id: str
        Frame id to publish the marker to
    
    Returns
    -------
    visualization_msgs.msg.Marker
        A marker message containing the given data        
    '''
    global colors
    
    msg = Marker()
    
    (x, y, z), (dx, dy, dz) = box_arr[:3], box_arr[3:6]
    (qw, qx, qy, qz) = Quaternion(axis=[0, 0, 1], angle=box_arr[6]).normalised.q
    
    # Construct the message header
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()
    msg.ns, msg.id = name_space, idx
    
    # Message type
    msg.type = Marker.CUBE
    msg.action = Marker.ADD
    
    # Box parameters
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = -z
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    msg.scale.x = dx
    msg.scale.y = dy
    msg.scale.z = dz
    
    # Box color
    RGB = colors[idx%100]  # Unique color using idx
    msg.color.r = RGB[0]
    msg.color.g = RGB[1]
    msg.color.b = RGB[2]
    msg.color.a = 0.75
    
    return msg
    
def del_msg(frame_id:str='') -> Marker:
    '''
    Create a marker with a delete message to delete all current markers
    
    Parameters
    ----------
    frame_id: str
        Frame id to publish the marker to
    
    Returns
    -------
    visualization_msgs.msg.Marker
        A marker message that deletes all current markers
    '''
    msg = Marker()
    
    # Marker header
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()
    
    # Marker type
    msg.type = Marker.CUBE
    msg.action = Marker.DELETEALL
    
    return msg
    
def visualize_bboxs(data:BBox3DArray):
    '''
    Plots the bounding boxes in the given data by publishing the corresponding MarkerArray that can be visualized in rviz
    
    Parameters
    ----------
    data: pcdet_ros_msgs.msg.BBox3DArray
        Contains all bounding boxes (must be tracked with unique ids)
    '''
    global counter
    
    frame_id = data.header.frame_id
    if override_frame_id is not None:
        frame_id = override_frame_id
    markers = [del_msg(frame_id)]  # Marker to delete all current existing Markers
    
    rviz_boxes_msg = MarkerArray()
    marker_ns = str(counter)
    counter += 1
    
    # Add a Marker for each box
    for box in data.boxes:
        box_id = box.box_id
        box_to_add = np.array([box.center.x, box.center.y, box.center.z, box.size.x, box.size.y, box.size.z, box.heading])
        unique_id = box.box_id
        markers.append(numpy_to_MarkerMsg(box_to_add, marker_ns, unique_id, frame_id))
        
    # Publish all markers
    rviz_boxes_msg.markers = markers
    rviz_boxes_pub.publish(rviz_boxes_msg)
    
if __name__ == "__main__":
    rospy.init_node("3D Bounding Box Plotter")
    
    # For plotting variables (Used as global variables)
    counter = 0
    colors = np.random.rand(100, 3)
    
    # Topic name parameters
    plotter_bboxes_topic = rospy.get_param("plotter_bboxes_topic")
    marker_tracking_topic = rospy.get_param("marker_tracking_topic")
    override_frame_id = rospy.get_param("override_frame_id",None)
    
    # Publishers and Subscribers
    rviz_boxes_pub = rospy.Publisher(marker_tracking_topic, MarkerArray, queue_size=100)
    rospy.Subscriber(plotter_bboxes_topic, BBox3DArray, callback=visualize_bboxs)
    
    rospy.spin()
