import numpy as np

from typing import Union, List
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker:
    '''
    This class represents the internal state of individual tracked objects observed as bbox.

    State = [x,y,z,heading,l,w,h,x_dot,y_dot_z_dot,heading_dot]
    Assumptions:
    - Shapes of bounding boxes are taken as constants (don't change over time), but the estimate is improved with new observations
    - Prediction model is a constant velocity model
    
    Attributes
    ----------
    kf: filterpy.kalman.KalmanFilter
        Kalman Filter used to track a bounding box
    id: int
        unique id given to this tracker
    history: list
    
    hits: int
        The number of times bounding boxes were associated with this tracker
    age: int
        The number of (frames) this tracker lasted. frames = number of calls to self.update(.) + number of calls to self.predict()
    time_since_update: int
        The number of frames since this tracker was lasted associated with a detected bounding box.
        Number of calls to self.predict() without calling self.update(.)
    count: static int
        Used to give each instantiation of the class a unique id
    '''
    count = 0  # Static variable used to give each instantiation of the class a unique id
    def __init__(self, bbox:Union[np.array, list]):
        '''
        Initialises a tracker using initial bounding box

        Parameters
        ----------
        bbox: np.array (or list) of floats, len=7
            Contains the initial bounding box observations, [x,y,z,heading,l,w,h]
        '''

        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=11, dim_z=7)  # dim_x is the shape of the internal state, dim_z is the shape of observations
        self.kf.F = np.eye(11)
        self.kf.F[0][7] = 1  # X_dot
        self.kf.F[1][8] = 1  # Y_dot
        self.kf.F[2][9] = 1  # Z_dot
        #self.kf.F[3][10] = 1  # theta_dot

        # Define measurement function
        self.kf.H = np.zeros((7, 11))
        self.kf.H[:7, :7] = np.eye(7)

        # Initialize uncertainties
        self.kf.R[3:6, 3:6] *= 10.
        self.kf.P[7:, 7:] *= 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[7:, 7:] *= 0.01

        # Initial state of bounding box
        self.kf.x[:7] = bbox.reshape(7, 1)

        # Unique id
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history = []
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def update(self,bbox:Union[np.array, list]):
        '''
        Updates the state vector with an observed bbox that has been associated to this tracker

        Parameters
        ----------
        bbox: np.array (or list) of floats, len=7
            Contains the initial bounding box observations, [x,y,z,heading,l,w,h]
        '''
        self.time_since_update = 0
        self.history = []
        self.hits += 1

        # Fix heading angle
        current_heading = self.kf.x[3]
        if bbox[3]-current_heading > 2*np.pi:
            bbox[3] -= 2*np.pi*np.floor((bbox[3]-current_heading)/(2*np.pi))
        elif bbox[3]-current_heading < 2*np.pi:
            bbox[3] -= 2*np.pi*np.ceil((bbox[3]-current_heading)/(2*np.pi))

        # Noise Removal, TODO: improve this
        if np.abs(current_heading-bbox[3]) > np.pi/2:
            bbox[3] = current_heading
            
        self.kf.update(bbox)

    def predict(self) -> Union[np.array, list]:
        '''
        Advances the state vector and returns the predicted bounding box estimated state

        Returns
        -------
        list of 11 floats: [x,y,z,heading,l,w,h,x_dot,y_dot,z_dot,heading_dot]
            The last estimated state of this tracker's bounding box
        '''
        self.kf.predict()
        
        self.age += 1
        self.time_since_update += 1
        self.history.append(self.kf.x)
        
        return self.kf.x

    def get_state(self) -> Union[np.array, list]:
        '''
        Returns the current bounding box's estimated state

        Returns
        -------
        list of 11 floats: [x,y,z,heading,l,w,h,x_dot,y_dot,z_dot,heading_dot]
            The last estimated state of this tracker's bounding box
        '''
        return self.kf.x
    
    def get_future_trajectories(self, num_steps:int=10) -> np.array:
        '''
        Moves the current bounding box using the estimated velocity a number of steps in the future and returns that predicted trajectory
        Note: it predicts num_steps into the future and doesn't include the bounding box's current state
        
        Parameters
        ----------
        num_steps: int
            The number of steps to predict into the future
        
        Returns
        -------
        np.array, shape = (num_steps, 7)
            Each row represents the bounding box state at the corresponding timestep
        '''
        predicted_states = []
        
        cur_state = self.get_state()
        forward_matrix = self.kf.F
        
        for i in range(num_steps):
            cur_state = np.dot(forward_matrix,cur_state)
            predicted_states.append(cur_state[:7])
        
        return np.array(predicted_states)
