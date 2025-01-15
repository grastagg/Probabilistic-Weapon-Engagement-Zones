import numpy as np
import matplotlib.pyplot as plt

from sacrificialAgent import sacrificialAgent
from pursuer import fastPursuer

# run main if main
if __name__ == "__main__":
    # initialize the sacrificial agent
    sacrificial = sacrificialAgent(1, np.array([0, 0, 0]))
    pursuerSpeed = 2
    pursuer = fastPursuer(pursuerSpeed, np.array([1, 1, 0]), 1, 0.1, type="proNav")
    pursuer2 = fastPursuer(
        pursuerSpeed, np.array([1, 1, 0]), 1, 0.1, type="proportional"
    )
    time = 0
    dt = 0.001
    endTime = 10
    speed = 1
    turnRate = 0.0
    poses = []
    pursuerPoses = []
    pursuer2Poses = []

    captured = pursuer.check_collision(sacrificial.pose)

    while time < endTime and not captured:
        pose = sacrificial.update(speed, turnRate, dt)
        pursuerPose = pursuer.update(pursuer.pose, dt, sacrificial.pose, speed)
        pursuer2Pose = pursuer2.update(pursuer2.pose, dt, sacrificial.pose, speed)
        poses.append(pose)
        pursuerPoses.append(pursuerPose)
        pursuer2Poses.append(pursuer2Pose)
        captured = pursuer2.check_collision(
            sacrificial.pose
        ) or pursuer.check_collision(sacrificial.pose)
        time += dt

    # plot the poses
    poses = np.array(poses)
    pursuerPoses = np.array(pursuerPoses)
    pursuer2Poses = np.array(pursuer2Poses)
    print(pursuerPoses)
    plt.plot(poses[:, 0], poses[:, 1], "b", label="Evader")
    plt.plot(pursuerPoses[:, 0], pursuerPoses[:, 1], "r", label=pursuer.type)
    plt.plot(pursuer2Poses[:, 0], pursuer2Poses[:, 1], "g", label=pursuer2.type)
    plt.legend()

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(pursuer.losHistory)), pursuer.losHistory)
    plt.show()
