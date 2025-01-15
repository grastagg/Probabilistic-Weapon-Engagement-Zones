import numpy as np


class sacrificialAgent:
    def __init__(self, speed, startPose):
        self.pose = startPose

    def update(self, speed, turnRate, dt):
        newPose = np.zeros(3)
        newPose[0] = self.pose[0] + speed * np.cos(self.pose[2]) * dt
        newPose[1] = self.pose[1] + speed * np.sin(self.pose[2]) * dt
        newPose[2] = self.pose[2] + turnRate * dt
        self.pose = newPose
        return newPose
