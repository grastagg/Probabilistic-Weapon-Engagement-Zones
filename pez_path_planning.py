import numpy as np

import matplotlib.pyplot as plt
from fast_pursuer import inEngagementZone, probabalisticEngagementZone, plotMalhalanobisDistance
from pyoptsparse import Optimization, OPT, IPOPT
from scipy.interpolate import BSpline
import time
from tqdm import tqdm




def create_spline(t0,tf,c,k,p0,pf,v0):
    '''
    :param t0: start and stop time of the spline (t0,t1)
    :param c: intermediate control points of the spline (3d controls points with shape (num_control_points-2,3) first and last control points are p0 and pf
    :param k: order of the spline
    :param p0: starting point of the spline (x,y,z)
    :param pf: ending point of the spline (x,y,z)
    :param v0: starting velocity of the spline (vx,vy,vz)
    :param vf: ennding velocity of the spline (vx,vy,vz)
    :return: scipy spline class with initial and final conditions set by parameters
    '''

    #### calculate the knot points of the spline ####
    c = c.reshape((-1,2))
    l = len(c) + 3 #number of control points is the number of intermediate control points plus the number of fixed control points


    t=np.linspace(t0,tf,l-k+1,endpoint=True)

    #initial knots for clamped bslpine
    t_0 = t0*np.ones(k)

    t=np.append(t_0, t)

    #final knots for clamped bspline
    t_f = tf * np.ones(k)
    t=np.append(t, t_f)

    #calculate the second control point
    c1 = v0 * t[k+1]/k + p0

    #stack all control points
    c_all = np.vstack((p0,c1,c,pf))
    # print(c_all)

    spline = BSpline(t,c_all,k)
    return spline
    
def compute_spline_constraints(spl, t, pursuerPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, useProbabalistic = True):
    pos = spl(t).squeeze()
    x = pos[:,0]
    y = pos[:,1]
    out_d1 = spl.derivative(1)(t).squeeze()
    out_d2 = spl.derivative(2)(t).squeeze()
    x1_dot = out_d1[:,0]
    x2_dot = out_d1[:,1]
    x1_ddot = out_d2[:,0]
    x2_ddot = out_d2[:,1]
    f_num = (np.multiply(x1_dot, x2_ddot) - np.multiply(x2_dot, x1_ddot))
    g_den = (np.square(x1_dot) + np.square(x2_dot))

    turn_rate = f_num / g_den

    velocity = np.sqrt(np.square(x1_dot) + np.square(x2_dot))

    curvature = turn_rate/velocity

    pez_constraint = np.zeros(len(t))
    for i in range(len(t)):
        agentHeading = np.arctan2(x2_dot[i], x1_dot[i])
        if useProbabalistic:
            pez_constraint[i] = probabalisticEngagementZone(np.array([[x[i]],[y[i]]]), agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
        else:
            pez_constraint[i] = inEngagementZone(np.array([[x[i]],[y[i]]]),agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
    

    
    


    return velocity, turn_rate, curvature, pez_constraint
  
def optimize_spline_path(p0, pf, v0, num_cont_points,spline_order, velocity_constraints, turn_rate_constraints, curvature_constraints, num_constraint_samples, pez_constraint_limit, pursuerPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, useProbabalistic):
    
    def objfunc(xdict):
        x_control_points = xdict["control_points"]
        tf = xdict["tf"]
        funcs = {}

        spline = create_spline(0, tf, x_control_points, spline_order, p0, pf, v0)

        
        t_constraints = np.linspace(0,tf, num_constraint_samples)

        velocity, turn_rate, curvature,pez = compute_spline_constraints(spline, t_constraints,pursuerPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, useProbabalistic)

        funcs['turn_rate'] = turn_rate
        funcs['velocity'] = velocity 
        funcs['curvature'] = curvature 
        funcs['pez'] = pez
        funcs['tf'] = tf 

        return funcs, False

    
    optProb = Optimization("path optimization", objfunc)

    x0_x = np.linspace(p0[0], pf[0], num_cont_points-3)
    x0_y = np.linspace(p0[0], pf[0], num_cont_points-3)
    x0 = np.hstack((x0_x,x0_y)).reshape((2*(num_cont_points-3)))

    optProb.addVarGroup(name="control_points", nVars=2*(num_cont_points - 3), varType="c", value=x0)
    optProb.addVarGroup(name="tf", nVars=1, varType="c", value=.1, lower=0,upper=None)

    optProb.addConGroup("velocity", num_constraint_samples, lower=velocity_constraints[0], upper=velocity_constraints[1], scale=1.0 / velocity_constraints[1])
    # optProb.addConGroup("turn_rate", num_constraint_samples, lower=turn_rate_constraints[0], upper=turn_rate_constraints[1], scale=1.0 / turn_rate_constraints[1])
    # optProb.addConGroup("curvature", num_constraint_samples, lower=curvature_constraints[0], upper=curvature_constraints[1], scale=1.0 / curvature_constraints[1])
    if useProbabalistic:
        optProb.addConGroup("pez", num_constraint_samples, lower=None, upper=pez_constraint_limit)
    else:
        optProb.addConGroup("pez", num_constraint_samples, lower=0, upper=None)



    optProb.addObj("tf")

    opt = OPT("ipopt")
    opt.options['print_level'] = 0
    opt.options['max_iter'] = 500

    sol = opt(optProb, sens="FD")
    # print(sol)

    print("time", sol.xStar['tf'])

    return create_spline(0, sol.xStar['tf'], sol.xStar['control_points'], spline_order, p0, pf, v0)

def plot_spline(spline, pursuerPosition, pursuerRange, pursuerCaptureRange,pez_limit,useProbabalistic):
    fig,ax = plt.subplots()
    t0 = spline.t[0]
    tf = spline.t[-1]
    t = np.linspace(t0, tf, 1000, endpoint=True)
    velocity, turn_rate, curvature,pez = compute_spline_constraints(spline, t,pursuerPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, useProbabalistic)
    if not useProbabalistic:
        spline_color = -pez
        plt.title("Basic Engagement Zone")
    else:
        spline_color = pez
        plt.title(f"Probabalistic Engagement Zone, PEZ Limit: {pez_limit}")
    #     print(spline_color)
    #     
    # first_constaint = np.nonzero(spline_color)[0][0]
    # last_constaint = np.nonzero(spline_color)[0][-1]
    x = spline(t)[:, 0]
    y = spline(t)[:, 1]
    # ax.plot(x[first_constaint-1:last_constaint+1], y[first_constaint-1:last_constaint+1], c = 'red')
    # ax.plot(x[0:first_constaint], y[0:first_constaint], c = 'green')
    # ax.plot(x[last_constaint:], y[last_constaint:], c = 'green')
    c = ax.scatter(x, y, c = spline_color, cmap = 'viridis',s=4)
    cbar = plt.colorbar(c)
    if useProbabalistic:
        cbar.set_label("Engagement Zone Probability")
        plotMalhalanobisDistance(pursuerPosition, pursuerPositionCov, ax)
    else:
        cbar.set_label("dist - rho")
    control_points = spline.c
    ax.plot(control_points[:, 0], control_points[:, 1], marker='o',linestyle = 'dashed',c = 'tab:gray')
    ax.set_aspect(1)
    c = plt.Circle(pursuerPosition, pursuerRange+pursuerCaptureRange, fill=False)
    plt.scatter(pursuerPosition[0], pursuerPosition[1], c='r')
    ax.add_artist(c)
    plt.xlabel("x")
    plt.ylabel("y")
    
def plot_constraints(spline, velocity_limits, turn_rate_limits, curvature_limits, pez_limit, useProbabalistic):
    fig, axs = plt.subplots(2)
    t = np.linspace(0, spline.t[-1], 100, endpoint=True)
    velocity, turn_rate, curvature,pez = compute_spline_constraints(spline, t,pursuerPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, useProbabalistic)

    axs[0].plot(t,velocity,c='b')
    axs[0].plot(t,np.ones(len(t))*velocity_limits[0],c='r')
    axs[0].plot(t,np.ones(len(t))*velocity_limits[1],c='r')
    axs[0].set_title("velocity")
    # axs[1].plot(t,turn_rate)
    # axs[1].plot(t,np.ones(len(t))*turn_rate_limits[0],c='r')
    # axs[1].plot(t,np.ones(len(t))*turn_rate_limits[1],c='r')
    # axs[1].set_title("turn rate")
    # axs[2].plot(t,curvature)
    # axs[2].plot(t,np.ones(len(t))*curvature_limits[0],c='r')
    # axs[2].plot(t,np.ones(len(t))*curvature_limits[1],c='r')
    # axs[2].set_title("curvature")

    axs[1].plot(t,pez)
    # axs[2].plot(t,np.ones(len(t))*curvature_limits[0],c='r')
    # axs[2].plot(t,np.ones(len(t))*curvature_limits[1],c='r')
    axs[1].set_title("Engagement Zone Probability")


def mc_spline_evaluation(spline, num_mc_runs, num_samples, pursuerPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed):
    t0 = spline.t[0]
    tf = spline.t[-1]
    t = np.linspace(t0, tf, num_samples, endpoint=True)
    num_bez_violations = 0
    for j in tqdm(range(num_mc_runs)):
        pursuerPositionTemp = np.random.multivariate_normal(pursuerPosition.squeeze(), pursuerPositionCov).reshape((-1,1))
        for i in range(num_samples):
            agentPosition = spline(t[i]).reshape((-1,1))
            agentHeading = np.arctan2(spline.derivative(1)(t[i])[1],spline.derivative(1)(t[i])[0])
            bez = inEngagementZone(agentPosition,agentHeading, pursuerPositionTemp, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
            if bez + 1e-8 < 0:
                num_bez_violations += 1
                break
                
        
    return num_bez_violations/num_mc_runs


if __name__ == '__main__':
    pursuerPosition = np.array([[0.0],[0.0]])
    pursuerPositionCov = np.array([[.1,0],[0,.1]])

    startingLocation = np.array([-4.0,-4.0])
    endingLocation = np.array([3.0,3.0])
    initialVelocity = np.array([1,1])
    numControlPoints = 7
    splineOrder = 3
    velocity_constraints = (0,10.0) 
    turn_rate_constraints = (-5.0,5.0) 
    curvature_constraints = (-1,1) 
    num_constraint_samples = 50
    # pez_constraint_limit_list = [.1,.2,.3,.4]
    pez_constraint_limit_list = [.1]
    pursuerRange = 2
    pursuerCaptureRange = 0.1
    pursuerSpeed = 1.0
    agentSpeed = .9

    num_mc_runs = 2000
    
    useProbabalistic = True

    for pez_constraint_limit in pez_constraint_limit_list:
        spline = optimize_spline_path(startingLocation, endingLocation, initialVelocity, numControlPoints,splineOrder, velocity_constraints, turn_rate_constraints, curvature_constraints, num_constraint_samples, pez_constraint_limit, pursuerPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, useProbabalistic)
        plot_constraints(spline, velocity_constraints, turn_rate_constraints, curvature_constraints, pez_constraint_limit, useProbabalistic)
        plot_spline(spline, pursuerPosition, pursuerRange, pursuerCaptureRange,pez_constraint_limit,useProbabalistic)
        bez_fail_percentage = mc_spline_evaluation(spline, num_mc_runs, num_constraint_samples, pursuerPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
        print("BEZ Fail Percentage: ", bez_fail_percentage)

    useProbabalistic = False
    spline = optimize_spline_path(startingLocation, endingLocation, initialVelocity, numControlPoints,splineOrder, velocity_constraints, turn_rate_constraints, curvature_constraints, num_constraint_samples, pez_constraint_limit, pursuerPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, useProbabalistic)
    plot_constraints(spline, velocity_constraints, turn_rate_constraints, curvature_constraints, pez_constraint_limit,useProbabalistic)
    plot_spline(spline, pursuerPosition, pursuerRange, pursuerCaptureRange,pez_constraint_limit, useProbabalistic)

    bez_fail_percentage = mc_spline_evaluation(spline, num_mc_runs, num_constraint_samples, pursuerPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
    print("BEZ Fail Percentage: ", bez_fail_percentage)
    
    plt.show()