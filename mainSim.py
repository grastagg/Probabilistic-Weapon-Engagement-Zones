import numpy as np
import matplotlib.pyplot as plt

from sacrificialAgent import sacrificialAgent
from pursuer import fastPursuer
from testDubins import (
    find_dubins_path_length_no_jax,
    new_in_dubins_engagement_zone_single,
    new_in_dubins_engagement_zone_single,
)

# run main if main
if __name__ == "__main__":
    # initialize the sacrificial agent
    # [-0.14408818 -0.25      ]
    #
    agentSpeed = 1
    agentInitialPosition = np.array([0.25, 0.4, 0])
    sacrificial = sacrificialAgent(agentSpeed, agentInitialPosition)
    pursuerSpeed = 2
    pursuerInitialPosition = np.array([0, 0, np.pi / 2])
    pursuerRange = 1
    pursuerCaptureRadis = 0.1
    pursuerTurnRadius = 0.5
    pursuer = fastPursuer(
        pursuerSpeed,
        pursuerInitialPosition,
        pursuerRange,
        pursuerCaptureRadis,
        type="collision",
        turnRadius=pursuerTurnRadius,
    )

    inEZ = new_in_dubins_engagement_zone_single(
        pursuerInitialPosition[0:2],
        pursuerInitialPosition[2],
        pursuerTurnRadius,
        pursuerCaptureRadis,
        pursuerRange,
        pursuerSpeed,
        sacrificial.pose[0:2],
        sacrificial.pose[2],
        agentSpeed,
    )
    print("In EZ: ", inEZ)

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
        captured = pursuer.check_collision(sacrificial.pose)
        pursuerRangeCheck = pursuer.check_range()
        time += dt
    print("Captured: ", captured)
    print("max range: ", not pursuerRangeCheck)

    fig, ax = plt.subplots()
    length = find_dubins_path_length_no_jax(
        pursuerInitialPosition[0:2],
        pursuerInitialPosition[2],
        pursuer.targetPoint,
        pursuerTurnRadius,
        ax,
    )
    # plot the poses
    poses = np.array(poses)
    pursuerPoses = np.array(pursuerPoses)
    plt.plot(poses[:, 0], poses[:, 1], "b", label="Evader")
    plt.plot(pursuerPoses[:, 0], pursuerPoses[:, 1], "r", label=pursuer.type)
    plt.legend()
    ax = plt.gca()
    ax.set_aspect("equal")

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(pursuer.losHistory)), pursuer.losHistory)
    plt.show()
