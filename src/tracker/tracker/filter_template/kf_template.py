import numpy as np
import scipy
from ..utils.tracker_utils import *

class KFTemplate(object):
    def __init__(self, initial_state):
        self.initial_state = initial_state

        self.init_matrix()
    
    def init_matrix(self):
        # H : 측정값 matrix (state 중 어떤 것이 측정값으로 들어오는지)
        '''     x   y   z   yaw vx  vy  w
        x       1   0   0   0   0   0   0
        y       0   1   0   0   0   0   0
        z       0   0   1   0   0   0   0
        yaw     0   0   0   1   0   0   0
        '''
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0]])
        
        # R : measurement 오차 공분산 행렬 (!!!튜닝 필요!!!)
        self.R = np.diag([0.001, 0.001, 1.0, 0.000001])
        
        # Q : system model 오차 공분산 행렬 (!!!튜닝 필요!!!)
        self.Q = np.diag([0.3, 0.3, 1.0, 0.05, 7.5, 7.5, 0.000001])

        self.P = np.eye(7)
        self.x = self.initial_state.reshape(-1, 1)
    
    def predict(self, dt):
        A = np.array([[1,  0,  0,  0, dt,  0,  0],
                      [0,  1,  0,  0,  0, dt,  0],
                      [0,  0,  1,  0,  0,  0,  0],
                      [0,  0,  0,  1,  0,  0, dt],
                      [0,  0,  0,  0,  1,  0,  0],
                      [0,  0,  0,  0,  0,  1,  0],
                      [0,  0,  0,  0,  0,  0,  1]])
        
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + self.Q

        return self.x
    
    def update(self, z):
        z = z.reshape(-1, 1)
        # z = check_opposite(z, self.x)
        
        K = self.P @ self.H.T @ scipy.linalg.inv(self.H @ self.P @ self.H.T + self.R)

        measurement_error = z - self.H @ self.x
        measurement_error[3,0] = angle_filter(measurement_error[3,0])

        self.x = self.x + K @ measurement_error
        self.x[2,0] = z[2,0]
        self.P = self.P - K @ self.H @ self.P

        return self.x

class RuleBasedKFTemplate(object):
    def __init__(self, initial_state):
        self.initial_state = initial_state

        self.init_matrix()
    
    def init_matrix(self):
        # H : 측정값 matrix (state 중 어떤 것이 측정값으로 들어오는지)
        '''     x   y   z   yaw vx  vy  w
        x       1   0   0   0   0   0   0
        y       0   1   0   0   0   0   0
        yaw     0   0   0   1   0   0   0
        '''
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0]])
        
        # R : measurement 오차 공분산 행렬 (!!!튜닝 필요!!!)
        self.R = np.diag([0.01, 0.01])
        
        # Q : system model 오차 공분산 행렬 (!!!튜닝 필요!!!)
        self.Q = np.diag([2.5, 2.5, 1.0, 0.05, 50.0, 50.0, 0.000001])

        self.P = np.eye(7)
        self.x = self.initial_state.reshape(-1, 1)
    
    def predict(self, dt):
        A = np.array([[1,  0,  0,  0, dt,  0,  0],
                      [0,  1,  0,  0,  0, dt,  0],
                      [0,  0,  1,  0,  0,  0,  0],
                      [0,  0,  0,  1,  0,  0, dt],
                      [0,  0,  0,  0,  1,  0,  0],
                      [0,  0,  0,  0,  0,  1,  0],
                      [0,  0,  0,  0,  0,  0,  1]])
        
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + self.Q

        return self.x
    
    def update(self, z):
        z = z.reshape(-1, 1)
        # z = check_opposite(z, self.x)
        
        K = self.P @ self.H.T @ scipy.linalg.inv(self.H @ self.P @ self.H.T + self.R)

        measurement_error = z - self.H @ self.x

        self.x = self.x + K @ measurement_error
        self.P = self.P - K @ self.H @ self.P

        return self.x
    

class StaticKFTemplate(object):
    def __init__(self, initial_state):
        self.initial_state = initial_state

        self.init_matrix()
    
    def init_matrix(self):
        # H : 측정값 matrix (state 중 어떤 것이 측정값으로 들어오는지)
        '''     x   y   z   yaw vx  vy  w
        x       1   0   0   0   0   0   0
        y       0   1   0   0   0   0   0
        '''
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],])
        
        # R : measurement 오차 공분산 행렬 (!!!튜닝 필요!!!)
        self.R = np.diag([1.0, 1.0])
        
        # Q : system model 오차 공분산 행렬 (!!!튜닝 필요!!!)
        self.Q = np.diag([10.0, 10.0, 1.0, 0.001, 10000.0, 10000.0, 0.000001])

        self.P = np.eye(7)
        self.x = self.initial_state.reshape(-1, 1)
    
    def predict(self, dt):
        A = np.array([[1,  0,  0,  0,  0,  0,  0],
                      [0,  1,  0,  0,  0,  0,  0],
                      [0,  0,  1,  0,  0,  0,  0],
                      [0,  0,  0,  1,  0,  0,  0],
                      [0,  0,  0,  0,  1,  0,  0],
                      [0,  0,  0,  0,  0,  1,  0],
                      [0,  0,  0,  0,  0,  0,  1]])
        
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + self.Q

        return self.x
    
    def update(self, z):
        z = z.reshape(-1, 1)
        # z = check_opposite(z, self.x)
        
        K = self.P @ self.H.T @ scipy.linalg.inv(self.H @ self.P @ self.H.T + self.R)

        measurement_error = z - self.H @ self.x
        # measurement_error[3,0] = angle_filter(measurement_error[3,0])

        self.x = self.x + K @ measurement_error
        self.P = self.P - K @ self.H @ self.P

        return self.x


class BBoxKFTemplate(object):
    def __init__(self, initial_state):
        self.initial_state = initial_state

        # H : measurement matrix
        '''     x1  y1  s   r   vx  vy  dr
        x1      1   0   0   0   0   0   0
        y1      0   1   0   0   0   0   0
        s       0   0   1   0   0   0   0
        r       0   0   0   1   0   0   0
        '''
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0]])

        self.R = np.eye(4)

        self.Q = np.eye(7)*0.5

        self.P = np.eye(7)*10
        self.P[4:, 4:] *= 10000

        self.x = np.zeros((7, 1))
        self.x[:4] = convert_bbox_to_z(self.initial_state)

    def predict(self, dt):
        A = np.array([[1, 0, 0, 0, dt, 0, 0],
                      [0, 1, 0, 0, 0, dt, 0],
                      [0, 0, 1, 0, 0, 0, dt],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1]])

        self.x = A @ self.x
        self.P = A @ self.P @ A.T + self.Q

        return self.x

    def update(self, z):
        z = z.reshape(-1, 1)

        K = self.P @ self.H.T @ scipy.linalg.inv(self.H @ self.P @ self.H.T + self.R)

        measurement_error = z - self.H @ self.x

        self.x = self.x + K @ measurement_error
        self.P = self.P - K @ self.H @ self.P

        return self.x

