import time
import rospy

import numpy as np

from visualization_msgs.msg import MarkerArray, Marker
from pcdet_ros_msgs.msg import BoundingBox3D,BoundingBoxes3D

import SORT3D as SORT_Tracker

    
def track_objects(data:BoundingBoxes3D):
    '''
    Tracks objects either using SORT by adding a unique id to each object detected
    
    Parameters
    ----------
    data: pcdet_ros_msgs.msg.BoundingBoxes3D
        The bounding boxes predicted from PV-RCNN (doesn't need the unique_id parameter to be set for each bounding box)
    '''
    global trackers
    
    tic = time.time()
    
    filtered_boxes = BoundingBoxes3D()
    filtered_boxes.header = data.header
    
    # Parse given BBoxes3D
    detections = [[] for i in range(num_classes)]
    for box in data.bounding_boxes:
        box_id = box.label
        if box_id > -1 and box_id < num_classes:
            box_to_add = [box.x, box.y, box.z, box.heading, box.dx, box.dy, box.dz]
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
                bbx = BoundingBox3D()
                bbx.x, bbx.y, bbx.z, bbx.heading, bbx.dx, bbx.dy, bbx.dz = np.array(f[:7], dtype=float)
                bbx.unique_id = int(f[-1])
                filtered_boxes.bounding_boxes.append(bbx)
    
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
    tracked_boxes = rospy.Publisher(tracked_objects_topic, BoundingBoxes3D, queue_size=1)
    rospy.Subscriber(object_detection_topic, BoundingBoxes3D, callback=track_objects)
    
    rospy.spin()
