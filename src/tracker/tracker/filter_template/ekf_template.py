import numpy as np
import scipy

class EKFTemplate(object):
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.last_result_state = np.zeros((7,))

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
        self.R = np.array([1.0, 1.0, 1.0, 0.000001])
        
        # Q : system model 오차 공분산 행렬 (!!!튜닝 필요!!!)
        self.Q = np.diag([10.0, 10.0, 1.0, 0.05, 50.0, 50.0, 0.000001])

        self.P = np.eye(7)
        self.P_prev = self.P
        self.x_prev = self.initial_state.reshape(-1, 1)
        self.x = self.x_prev
    

    def f(self, x_prev, dt):
        x = x_prev
        # x[0] += x_prev[4]*dt # x
        # x[1] += x_prev[5]*dt # y
        x[0] += x_prev[4]*dt*np.cos(x_prev[6]*dt)-x_prev[5]*dt*np.sin(x_prev[6]*dt) # x
        x[1] += x_prev[4]*dt*np.sin(x_prev[6]*dt)+x_prev[5]*dt*np.cos(x_prev[6]*dt) # y
        x[3] = np.arctan2(x[5],x[4]) + x_prev[6]*dt # yaw

        return x
    

    def Ajacob(self, x_prev, dt):
        A = np.eye(7)

        # A[0,4] = dt
        # A[1,5] = dt
        # A[3,3] = 0

        A[0,4] = dt*np.cos(x_prev[6]*dt)
        A[0,5] = -dt*np.sin(x_prev[6]*dt)
        A[0,6] = -x_prev[4]*(dt**2)*np.sin(x_prev[6]*dt)-x_prev[5]*(dt**2)*np.cos(x_prev[6]*dt)

        # y
        A[1,4] = dt*np.sin(x_prev[6]*dt)
        A[1,5] = dt*np.cos(x_prev[6]*dt)
        A[1,6] = x_prev[4]*(dt**2)*np.cos(x_prev[6]*dt)-x_prev[5]*(dt**2)*np.sin(x_prev[6]*dt)
        
        # yaw
        A[3,3] = 0
        A[3,4] = -x_prev[5] / (x_prev[4]**2 + x_prev[5]**2)
        A[3,5] = x_prev[4] / (x_prev[4]**2 + x_prev[5]**2)
        A[3,6] = dt

        return A


    def predict(self, dt):
        # state 예측
        self.x = self.f(self.x, dt)
        
        # 오차 공분산 행렬 예측
        Ajacob = self.Ajacob(self.x, dt)
        self.P = Ajacob @ self.P @ Ajacob.T + self.Q

        return self.x


    def update(self, z):
        z = z.reshape(-1, 1)
        
        K = self.P @ self.H.T @ scipy.linalg.inv(self.H @ self.P @ self.H.T + self.R)

        measurement_error = z - self.H @ self.x

        self.x = self.x + K @ measurement_error
        self.P = self.P - K @ self.H @ self.P

        return self.x
    

class RuleBasedEKFTemplate(object):
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.last_result_state = np.zeros((7,))

        self.init_matrix()

    
    def init_matrix(self):
        # H : 측정값 matrix (state 중 어떤 것이 측정값으로 들어오는지)
        '''     x   y   z   yaw vx  vy  w
        x       1   0   0   0   0   0   0
        y       0   1   0   0   0   0   0
        '''
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0]])
        
        # R : measurement 오차 공분산 행렬 (!!!튜닝 필요!!!)
        self.R = np.diag([1.0, 1.0])
        
        # Q : system model 오차 공분산 행렬 (!!!튜닝 필요!!!)
        self.Q = np.diag([10.0, 10.0, 1.0, 10.0, 100.0, 100.0, 10.0])

        self.P = np.eye(7)
        self.x = self.initial_state.reshape(-1,1)
    

    def f(self, x_prev, dt):
        x = x_prev
        # x[0] += x_prev[4]*dt # x
        # x[1] += x_prev[5]*dt # y

        x[0] += x_prev[4]*dt*np.cos(x_prev[6]*dt)-x_prev[5]*dt*np.sin(x_prev[6]*dt) # x
        x[1] += x_prev[4]*dt*np.sin(x_prev[6]*dt)+x_prev[5]*dt*np.cos(x_prev[6]*dt) # y

        x[3] = np.arctan2(x[5],x[4]) + x_prev[6]*dt # yaw

        return x
    

    def Ajacob(self, x_prev, dt):
        A = np.eye(7)

        # A[0,4] = dt # x
        # A[1,5] = dt # y

        # x
        A[0,4] = dt*np.cos(x_prev[6]*dt)
        A[0,5] = -dt*np.sin(x_prev[6]*dt)
        A[0,6] = -x_prev[4]*(dt**2)*np.sin(x_prev[6]*dt)-x_prev[5]*(dt**2)*np.cos(x_prev[6]*dt)

        # y
        A[1,4] = dt*np.sin(x_prev[6]*dt)
        A[1,5] = dt*np.cos(x_prev[6]*dt)
        A[1,6] = x_prev[4]*(dt**2)*np.cos(x_prev[6]*dt)-x_prev[5]*(dt**2)*np.sin(x_prev[6]*dt)
        
        # yaw
        A[3,3] = 0
        A[3,4] = -x_prev[5] / (x_prev[4]**2 + x_prev[5]**2)
        A[3,5] = x_prev[4] / (x_prev[4]**2 + x_prev[5]**2)
        A[3,6] = dt

        return A


    def predict(self, dt):
        # state 예측
        self.x = self.f(self.x, dt)
        
        # 오차 공분산 행렬 예측
        Ajacob = self.Ajacob(self.x, dt)
        self.P = Ajacob @ self.P @ Ajacob.T + self.Q

        return self.x


    def update(self, z):
        z = z.reshape(-1, 1)
        
        K = self.P @ self.H.T @ scipy.linalg.inv(self.H @ self.P @ self.H.T + self.R)

        measurement_error = z - self.H @ self.x

        self.x = self.x + K @ measurement_error
        self.P = self.P - K @ self.H @ self.P

        return self.x