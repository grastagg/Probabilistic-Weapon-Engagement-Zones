import numpy as np
import matplotlib.pyplot as plt

from sacrificialAgent import sacrificialAgent
from pursuer import fastPursuer

# run main if main
if __name__ == "__main__":
    # initialize the sacrificial agent
    agentSpeed = 1
    agentInitialPosition = np.array([-0.5, 0.9, 0])
    sacrificial = sacrificialAgent(agentSpeed, agentInitialPosition)
    pursuerSpeed = 4
    pursuerInitialPosition = np.array([0, 0, np.pi / 2])
    pursuerRange = 1
    pursuerCaptureRadis = 0.1
    pursuer = fastPursuer(
        pursuerSpeed,
        pursuerInitialPosition,
        pursuerRange,
        pursuerCaptureRadis,
        type="proportional",
    )
    time = 0
    dt = 0.001
    endTime = 10
    speed = 1
    turnRate = 0.0
    poses = []
    pursuerPoses = []

    captured = pursuer.check_collision(sacrificial.pose)
    pursuerRangeCheck = pursuer.check_range()

    while time < endTime and not captured and pursuerRangeCheck:
        pose = sacrificial.update(speed, turnRate, dt)
        pursuerPose = pursuer.update(pursuer.pose, dt, sacrificial.pose, speed)
        poses.append(pose)
        pursuerPoses.append(pursuerPose)
        pursuer.check_collision(sacrificial.pose)
        pursuerRangeCheck = pursuer.check_range()
        time += dt
    print("Captured: ", captured)
    print("max range: ", not pursuerRangeCheck)

    # plot the poses
    poses = np.array(poses)
    pursuerPoses = np.array(pursuerPoses)
    plt.plot(poses[:, 0], poses[:, 1], "b", label="Evader")
    plt.plot(pursuerPoses[:, 0], pursuerPoses[:, 1], "r", label=pursuer.type)
    plt.legend()

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(pursuer.losHistory)), pursuer.losHistory)
    plt.show()
