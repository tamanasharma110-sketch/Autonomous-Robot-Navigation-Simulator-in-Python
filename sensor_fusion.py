import numpy as np

class KalmanFilter:
    """Fuses noisy GPS + IMU readings into a clean pose estimate."""
    def __init__(self):
        self.state = np.zeros(4)          # [x, y, vx, vy]
        self.P = np.eye(4) * 1000        # uncertainty
        self.Q = np.eye(4) * 0.01        # process noise
        self.R = np.eye(2) * 5.0         # measurement noise (GPS)

    def predict(self, dt):
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

    def update(self, gps_measurement):
        H = np.array([[1,0,0,0],[0,1,0,0]])
        y = gps_measurement - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.state[:2]  # return fused x, y