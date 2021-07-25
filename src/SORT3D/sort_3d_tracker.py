import numpy as np

from typing import Union, List, Tuple
from scipy.optimize import linear_sum_assignment

from .KalmanBoxTracker import *


def linear_assignment(cost_matrix:Union[np.array, List[list]]) -> np.array:
    '''
    Solves the linear assigment problem using the hungarian algorithm using "scipy"
    
    Example:
    >>> linear_assignment(np.array([[1,2],[3,4]]))
    array([[0, 0],
           [1, 1]])
    
    Parameters
    ----------
    cost_matrix: nxm numpy array of floats
        Symmetric matrix, each element x_ij represents the cost of assigning i to j
    
    Returns
    -------
    numpy array of shape (min(n,m),2)
        Each row is an assignment of a row to a column
        Ex: [1,2] means row 1 should be assigned to the element in row 1 column 2
    '''
    x, y = linear_sum_assignment(cost_matrix,maximize=False)
    return np.array(list(zip(x, y)))


def euclidean_mat(bb_test:np.array, bb_gt:np.array) -> np.array:
    '''
    Currently will use eulclidean distance between centers
    Note: Could add distance using theta and scale later
    
    Parameters
    ----------
    bb_test: np.array of floats, shape = (N,7)
        Each row represents a bounding box [x,y,z,heading,l,w,h]
    bb_gt: np.array of floats, shape = (M,7)
        Each row represents a bounding box [x,y,z,heading,l,w,h]
    
    Returns
    -------
    np.array of floats, shape=(N,M)
        Each element x_ij is the euclidean distance between the center of bb_test[i] and bb_gt[j]
    '''
    N, M = bb_test.shape[0], bb_gt.shape[0]
    
    bb_test_arr = np.ones((N,M,3))*(bb_test[:,:3].reshape(-1,1,3))
    bb_gt_arr = np.ones((N,M,3))*(bb_gt[:,:3].reshape(1,-1,3))
    
    diff = bb_test_arr-bb_gt_arr
    
    output = np.linalg.norm(diff,axis=2)
            
    return output

def pi2pi(angle: Union[float,int,np.array]) -> Union[float,int,np.array]:
    '''
    Converts an angle to -pi to pi format
    
    Parameters
    ----------
    angle: float
        The angle to convert (in radians)
    
    Returns
    -------
    float: given angle converted to an angle between -pi to pi (radians)
    '''
    return angle%(2*np.pi) - np.pi


def associate_detections_to_trackers(detections:Union[np.array, List[list]], 
                                      trackers:List[KalmanBoxTracker], 
                                      dist_threshold:float=2.0) -> Tuple[np.array, np.array, np.array]:
    '''
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers

    Parameters
    ----------
    detections: np.array of floats, shape = (N,7)
        Each row contains a detected bounding box [x,y,z,heading,l,w,h]
    trackers: list of KalmanBoxTracker
        List containing all the trackers of previously detected bounding boxes
    dist_threshold: float
        Maximum distance where 2 boxes can be associated
        
    Returns
    -------
    matches: np.array, shape = (num_matches,2)
        Each row contains 2 ints [idx1, idx2], that represent matching idx1 from detections to idx2 from trackers
    
    unmatched_detections: np.array, shape = (num_unmatched_detections, )
        Each element represents an index of an element from the detections array that wasn't matched with any trackers
    
    unmatched_trackers: np.array, shape = (num_unmatched_trackers,)
        Each element represents an index of an element from the trackers array that wasn't matched with any detections
    '''
    if(len(trackers)==0):
        return np.empty((0, 2),dtype=np.float), np.arange(len(detections)), np.empty((0, 7), dtype=np.float)

    costs_matrix = euclidean_mat(detections, trackers)

    if min(costs_matrix.shape) > 0:
        a = (costs_matrix < dist_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(costs_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    
    # Filter out matched with too large distance
    matches = []
    for m in matched_indices:
        if(costs_matrix[m[0], m[1]] > dist_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            
    if len(matches)==0:
        matches = np.empty((0, 2), dtype=np.float)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)


class SORT3D:
    '''
    Main SORT3D class to manage all trackers and data associations
    
    Attributes
    ----------
    max_age: int
        The maximum number of frames a bounding box can survive without detections before being eliminated
    min_hits: int
        The minimum number of hits on a bounding box before it is returns as a tracked object
    dist_threshold: float
        Maximum distance where 2 boxes can be associated
    trackers: list
        A list of KalmanBoxTracker, each one tracks an object/bounding box
    frame_count: int
        The number of frames the SORT tracker has been up for
    '''
    def __init__(self, max_age:int=3, min_hits:int=3, dist_threshold:float=2.0):
        '''
        Sets key parameters for SORT
        '''
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 7))):
        '''
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 7)) for frames without detections)

        Parameters
        ----------
        dets: np.array, shape = (N,2)
            Each row is one detection represented as [x,y,z,heading,l,w,h]
          
        Returns
        -------
        a similar array, where the last column is the object ID.
        '''
        self.frame_count += 1
        
        # Get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict().T[0]
            trks[t] = pos[:7]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks.pop(t)
        
        # Do data associations
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.dist_threshold)

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
            
        to_return = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            trk_state = trk.get_state().reshape(-1)[:7]
            if trk.time_since_update <= self.max_age and trk.hits >= self.min_hits:
                to_add = np.concatenate([trk_state, [trk.id]])  # Add id to the bounding box state
                to_return.append(to_add)
            i -= 1
            
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        return np.array(to_return)  # Shape = (-1,8)
