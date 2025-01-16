import numpy as np


class fastPursuer:
    def __init__(self, speed, startPose, range, captureRadius, type="proportional"):
        self.pose = startPose
        self.speed = speed
        self.type = type
        self.range = range
        self.captureRadius = captureRadius

        turnRadus = 0.5
        self.maxTurnRate = speed / turnRadus

        self.distanceTravelled = 0

        self.losHistory = []

    def update(self, pose, dt, targetPose, targetSpeed):
        newPose = np.zeros(3)
        omega = 0
        if self.type == "proNav":
            omega = self.pronav_control(pose, targetPose, targetSpeed)
        if self.type == "proportional":
            omega = self.proportional_guidance_control(pose, targetPose)
        omega = np.clip(omega, -self.maxTurnRate, self.maxTurnRate)
        newPose[0] = pose[0] + self.speed * np.cos(pose[2]) * dt
        newPose[1] = pose[1] + self.speed * np.sin(pose[2]) * dt
        newPose[2] = pose[2] + omega * dt
        self.pose = newPose
        self.distanceTravelled += self.speed * dt

        return newPose

    def check_range(self):
        return self.distanceTravelled < self.range

    def proportional_guidance_control(self, pose, targetPose):
        angleToTarget = np.arctan2(targetPose[1] - pose[1], targetPose[0] - pose[0])
        angleDifference = angleToTarget - pose[2]
        angleDifference = np.arctan2(np.sin(angleDifference), np.cos(angleDifference))
        gain = 10.0
        omega = gain * angleDifference
        return omega

    def pronav_control(self, pose, targetPose, targetSpeed):
        perpenAcc, omega = self.calculate_los(pose, targetPose, self.speed, targetSpeed)
        turnRate = (
            10 * (self.speed / np.linalg.norm(targetPose[0:2] - pose[0:2])) * omega
        )
        # omega = gain * (self.speed / distance) * losRate
        # omega = gain * losRate

        return turnRate

    # def LOS_rate(self, pose, targetPose, targetSpeed):
    #     """
    #     Computes the rate of change of the line-of-sight angle (LOS rate).
    #     """
    #     x_p, y_p, theta_p = pose
    #     x_t, y_t, theta_t = targetPose
    #     v_p = self.speed
    #     v_t = targetSpeed
    #
    #     dx = x_t - x_p
    #     dy = y_t - y_p
    #     relative_velocity = (v_t - v_p) * np.array(
    #         [np.cos(theta_t), np.sin(theta_t)]
    #     ) - v_p * np.array([np.cos(theta_p), np.sin(theta_p)])
    #     LOS_angle = np.arctan2(dy, dx)
    #     self.losHistory.append(LOS_angle)
    #     LOS_rate = (relative_velocity[0] * dy - relative_velocity[1] * dx) / (
    #         dx**2 + dy**2
    #     )
    #     return LOS_rate
    # Function to calculate LOS angle and its rate of change (angular velocity)
    def calculate_los(self, missilePose, targetPose, missileSpeed, targetSpeed):
        # Calculate relative position vector
        relativePosition = targetPose[0:2] - missilePose[0:2]

        targetVelosity = targetSpeed * np.array(
            [np.cos(targetPose[2]), np.sin(targetPose[2])]
        )
        missileVelosity = missileSpeed * np.array(
            [np.cos(missilePose[2]), np.sin(missilePose[2])]
        )

        # Calculate relative velocity (difference in velocity)
        relativeVelocity = targetVelosity - missileVelosity

        relativePosition = np.array([relativePosition[0], relativePosition[1], 0])
        relativeVelocity = np.array([relativeVelocity[0], relativeVelocity[1], 0])
        omega = (np.cross(relativePosition, relativeVelocity)) / (
            np.dot(relativePosition, relativePosition)
        )

        gain = 2
        perpendicularAcceleration = gain * np.cross(relativeVelocity, omega)

        omega = omega[2]
        perpendicularAcceleration = perpendicularAcceleration[0:2]

        losAngle = np.arctan2(relativePosition[1], relativePosition[0])
        self.losHistory.append(losAngle)

        return perpendicularAcceleration, omega

    def check_collision(self, targetPose):
        dist = np.linalg.norm(self.pose[0:2] - targetPose[0:2])
        if dist < self.captureRadius:
            return True
        return False
