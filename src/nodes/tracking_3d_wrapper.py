#!/usr/bin/env python3


import time
import rospy

import numpy as np

from visualization_msgs.msg import MarkerArray, Marker
from arl_msgs.msg import BBox3D, BBox3DArray

import SORT3D as SORT_Tracker

    
def track_objects(data:BBox3D):
    '''
    Tracks objects either using SORT by adding a unique id to each object detected
    
    Parameters
    ----------
    data: arl_msgs.msg.BBox3DArray
        The bounding boxes predicted from PV-RCNN (doesn't need the unique_id parameter to be set for each bounding box)
    '''
    global trackers
    
    tic = time.time()
    
    filtered_boxes = BBox3DArray()
    filtered_boxes.header = data.header
    
    # Parse given BBoxes3D
    detections = [[] for i in range(num_classes)]
    for box in data.boxes:
        box_id = box.box_id
        if box_id > -1 and box_id < num_classes:
            box_to_add = [box.center.x, box.center.y, box.center.z, box.heading, box.size.x, box.size.y, box.size.z]
            detections[box_id].append(box_to_add)
        else:
            rospy.logwarn("Class {} is unidentified. It will be ignored by the tracker.".format(box.class_id))
            
    # Update trackers, and add them to message
    for idx, dets in enumerate(detections):
        to_track = []
        if len(dets) > 0:
            to_track = trackers[idx].update(np.array(dets))
        else:
            to_track = trackers[idx].update()

        # Add filtered cones to message that will be published, along with a unique id
        if len(to_track) > 0:
            for f in to_track:
                bbx = BBox3D()
                bbx.center.x, bbx.center.y, bbx.center.z, bbx.heading, bbx.size.x, bbx.size.y, bbx.size.z = np.array(f[:7], dtype=float)
                bbx.box_id = int(f[-1])
                filtered_boxes.boxes.append(bbx)
    
    toc = time.time()

    if log_info:
        rospy.loginfo("Finished Bounding Box filtering in {:.2} ms".format((toc - tic)*1000))

    # Publish filtered cones
    tracked_boxes.publish(filtered_boxes)


if __name__ == "__main__":
    rospy.init_node('3D SORT tracker')

    # Tracking method
    tracking_method = rospy.get_param("tracking_method", "SORT")

    # Algorithm parameters for SORT
    max_age = rospy.get_param("max_age", 3)
    min_hits = rospy.get_param("min_hits", 3)
    min_iou_thresh = rospy.get_param("min_iou_thresh", 2)
    num_classes = rospy.get_param("num_classes",80)
        
    # Topic name parameters
    object_detection_topic = rospy.get_param("object_detection_topic")
    tracked_objects_topic = rospy.get_param("tracked_objects_topic")

    # Logging parameters
    log_info = rospy.get_param("log_info", True)
    

    # Create a tracker for each class of objects
    trackers = []
    for i in range(num_classes):
        if tracking_method == "SORT":
            trackers.append(SORT_Tracker.SORT3D(max_age, min_hits, min_iou_thresh))
        else:
            rospy.logfatal("Unknown tracking method {}. Currently, can only use \"SORT\"".format(tracking_method))

    # Publishers and Subscribers
    tracked_boxes = rospy.Publisher(tracked_objects_topic, BBox3DArray, queue_size=10)
    rospy.Subscriber(object_detection_topic, BBox3DArray, callback=track_objects)
    
    rospy.spin()
