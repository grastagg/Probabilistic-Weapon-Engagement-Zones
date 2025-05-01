import jax
from jax._src.ad_checkpoint import saved_residuals
import tqdm
import os
import time
import datetime

import optax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jax import random, jit, vmap, grad, value_and_grad
from flax import serialization
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import qmc


import dubinsEZ
import dubinsPEZ
import mlp

in_dubins_engagement_zone_ev = jax.jit(
    jax.vmap(
        dubinsEZ.in_dubins_engagement_zone_single,
        in_axes=(
            None,  # pursuerPosition
            None,  # pursuerHeading
            None,  # minimumTurnRadius
            None,  # catureRadius
            None,  # pursuerRange
            None,  # pursuerSpeed
            0,  # evaderPosition
            0,  # evaderHeading
            None,  # evaderSpeed
        ),  # Vectorizing over evaderPosition & evaderHeading
    )
)

in_dubins_engagement_zone = jax.jit(
    jax.vmap(
        dubinsEZ.in_dubins_engagement_zone_single,
        in_axes=(
            0,  # pursuerPosition
            0,  # pursuerHeading
            0,  # minimumTurnRadius
            None,  # catureRadius
            0,  # pursuerRange
            0,  # pursuerSpeed
            0,  # evaderPosition
            0,  # evaderHeading
            None,  # evaderSpeed
        ),  # Vectorizing over evaderPosition & evaderHeading
    )
)

# mc_dubins_pez_vmap = jax.jit(
#     jax.vmap(
#         dubinsPEZ.mc_dubins_PEZ_Single,
#         in_axes=(
#             0,  # evaderPosition
#             0,  # evaderHeading
#             0,  # evaderSpeed
#             0,  # pursuerPosition
#             0,  # pursuerPositionCov
#             0,  # pursuerHeading
#             0,  # pursuerHeadingVar
#             0,  # pursuerSpeed
#             0,  # pursuerSpeedVar
#             0,  # pursuerTurnRadius
#             0,  # pursuerTurnRadiusVar
#             0,  # pursuerRange
#             0,  # pursuerRangeVar
#             None,
#         ),
#     )
# )


def mc_combined_input_single(X):
    pursuerPosition = jnp.array([0.0, 0.0])
    pursuerPositionCov = jnp.array([[X[0], X[2]], [X[2], X[1]]]) + jnp.eye(2) * 1e-6
    pursuerHeadingVar = X[3]
    minimumTurnRadius = X[4]
    minimumTurnRadiusVar = X[5]
    pursuerRange = X[6]
    pursuerRangeVar = X[7]
    pursuerSpeed = X[8]
    pursuerSpeedVar = X[9]
    evaderPosition = X[10:12]
    evaderHeading = X[12]
    evaderSpeed = X[13]
    pursuerHeading = 0.0

    p, _, _, _, _, _, _ = dubinsPEZ.mc_dubins_PEZ_Single(
        # p, e, _, _, _, _, _ = dubinsPEZ.mc_dubins_PEZ_Single_differentiable(
        evaderPosition,
        evaderHeading,
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        0.0,
    )
    z = dubinsEZ.in_dubins_engagement_zone_single(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        0.0,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )
    # zlinear, _, _ = dubinsPEZ.linear_dubins_PEZ_single(
    #     evaderPosition,
    #     evaderHeading,
    #     evaderSpeed,
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     0.0,
    # )
    # return zlinear - z
    #
    return p, z
    # eps = 1e-6
    # p = jnp.clip(z, eps, 1 - eps)
    # return jnp.log(p / (1 - p))  # logit transformation


def mc_combined_input_single_derivative(X):
    pursuerPosition = jnp.array([0, 0])
    pursuerPositionCov = jnp.array([[X[0], X[2]], [X[2], X[1]]])
    pursuerHeading = X[3]
    pursuerHeadingVar = X[4]
    minimumTurnRadius = X[5]
    minimumTurnRadiusVar = X[6]
    pursuerRange = X[7]
    pursuerRangeVar = X[8]
    pursuerSpeed = X[9]
    pursuerSpeedVar = X[10]
    # evaderPosition = jnp.array([X[11:13]])
    # evaderHeading = jnp.array([X[13]])
    evaderPosition = X[11:13]
    evaderHeading = X[13]
    evaderSpeed = X[14]
    z = dubinsPEZ.mc_dubins_PEZ_Single_differentiable(
        # z, _, _ = dubinsPEZ.dubins_pez_numerical_integration_sparse(
        evaderPosition,
        evaderHeading,
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        0.0,
        # dubinsPEZ.nodes,
        # dubinsPEZ.weights,
    )

    return z.squeeze()


mc_combined_input = jax.jit(jax.vmap(mc_combined_input_single, in_axes=(0,)))

mc_combined_input_derivative = jax.jit(
    jax.vmap(mc_combined_input_single_derivative, in_axes=(0,))
)
# mc_combined_input_jac = jax.jit(jax.vmap(mc_combined_input_jac_single, in_axes=(0,)))

mc_combined_input_vmap = jax.jit(jax.vmap(mc_combined_input_single, in_axes=(0,)))


def create_covariance_matrix_single(var1, var2, cov):
    return jnp.array([[var1, cov], [cov, var2]])


create_covariance_matrix = jax.jit(
    jax.vmap(create_covariance_matrix_single, in_axes=(0, 0, 0))
)


# -----------------------------------------------------------------------------
# 1) Latin‑Hypercube sampling for all input dims
# -----------------------------------------------------------------------------
def sample_inputs_lhs(
    n_samples: int,
    pursuerPositionCovMax: float,
    pursuerHeadingVarMax: float,
    pursuerTurnRadiusRange: tuple[float, float],
    pursuerTurnRadiusVarMax: float,
    pursuerRangeRange: tuple[float, float],
    pursuerRangeVarMax: float,
    pursuerSpeedRange: tuple[float, float],
    pursuerSpeedVarMax: float,
    evaderXPositionRange: tuple[float, float],
    evaderYPositionRange: tuple[float, float],
    evaderHeadingRange: tuple[float, float],
    evaderSpeedRange: tuple[float, float],
):
    """
    Returns individual arrays for each input parameter, sampled via LHS.
    """
    # total input dims (matching original create_input_output_data):
    # varX, varY, covXY,
    # heading, headingVar,
    # turnRadius, turnRadiusVar,
    # range, rangeVar,
    # speed, speedVar,
    # evaderX, evaderY, evaderHeading, evaderSpeed
    d = 14  # total input dims
    seed = np.random.randint(0, 1000000)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit = sampler.random(n=n_samples)  # shape (n_samples, 15)

    # helper for single-dim mapping
    def m1(u, lo, hi):
        return u * (hi - lo) + lo

    col = 0
    pvx = m1(unit[:, col], 0, pursuerPositionCovMax)
    col += 1
    pvy = m1(unit[:, col], 0, pursuerPositionCovMax)
    col += 1
    # for the covariance off-diagonal we still need a symmetric range:
    maxcov = np.sqrt(pvx * pvy)
    pcov = (unit[:, col] * 2 - 1) * maxcov
    col += 1
    phvar = m1(unit[:, col], 0, pursuerHeadingVarMax)
    col += 1
    ptrad = m1(unit[:, col], *pursuerTurnRadiusRange)
    col += 1
    ptvar = m1(unit[:, col], 0, pursuerTurnRadiusVarMax)
    col += 1
    prng = m1(unit[:, col], *pursuerRangeRange)
    col += 1
    prvar = m1(unit[:, col], 0, pursuerRangeVarMax)
    col += 1
    pspd = m1(unit[:, col], *pursuerSpeedRange)
    col += 1
    psvar = m1(unit[:, col], 0, pursuerSpeedVarMax)
    col += 1
    ex = m1(unit[:, col], *evaderXPositionRange)
    col += 1
    ey = m1(unit[:, col], *evaderYPositionRange)
    col += 1
    ehead = m1(unit[:, col], *evaderHeadingRange)
    col += 1
    espd = m1(unit[:, col], *evaderSpeedRange)
    col += 1
    # pack into X: same ordering expected by your unstack
    X = np.column_stack(
        [
            pvx,
            pvy,
            pcov,
            phvar,
            ptrad,
            ptvar,
            prng,
            prvar,
            pspd,
            psvar,
            ex,
            ey,
            ehead,
            espd,
        ]
    )  # shape (n_samples, 15)
    return X

    # # stack evader positions
    # epos = np.stack((ex, ey), axis=1)  # (n_samples,2)
    #
    # # create covariances: function create_covariance_matrix(varX, varY, covXY)
    # ppos_cov = create_covariance_matrix(pvx, pvy, pcov)  # (n_samples,2,2)
    #
    # return {
    #     "pvx": pvx,
    #     "pvy": pvy,
    #     "pcov": pcov,
    #     "phead": phead,
    #     "phvar": phvar,
    #     "ptrad": ptrad,
    #     "ptvar": ptvar,
    #     "prng": prng,
    #     "prvar": prvar,
    #     "pspd": pspd,
    #     "psvar": psvar,
    #     "epos": epos,
    #     "ehead": ehead,
    #     "espd": espd,
    #     "ppos_cov": ppos_cov,
    # }


# -----------------------------------------------------------------------------
# 2) Batched MC‑PEZ wrapper
# -----------------------------------------------------------------------------
def batched_mc_dubins_pez(
    X,
    batch_size=20,
):
    N = X.shape[0]
    ys = []
    ydot = []
    zs = []
    for i in tqdm.tqdm(range(0, N, batch_size)):
        j = i + batch_size
        X_batch = X[i:j]
        yb, z = mc_combined_input(jnp.array(X_batch))
        yd = mc_combined_input_derivative(jnp.array(X_batch))
        ys.append(yb)
        ydot.append(yd)
        zs.append(z)

    return (
        np.concatenate(ys, axis=0),
        np.concatenate(ydot, axis=0),
        np.concatenate(zs, axis=0),
    )  # shape (N,)


# -----------------------------------------------------------------------------
# 3) End‑to‑end data creation
# -----------------------------------------------------------------------------
def create_input_output_data_full(
    n_samples: int,
    # same args as sample_inputs_lhs...
    *ranges_and_maxes,
):
    # unpack all the args you passed through; for brevity assume same order
    X = sample_inputs_lhs(n_samples, *ranges_and_maxes)

    # run batched MC‑PEZ
    y, ydot, z = batched_mc_dubins_pez(
        X,
        batch_size=200,
    )
    # append z to xData
    # X = np.hstack((X, z[:, None]))
    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    ydot_train, ydot_test = train_test_split(ydot, test_size=0.2)
    # save train and test data
    xTrainDataFile = "data/xTrainData.csv"
    yTrainDataFile = "data/yTrainData.csv"
    ydotTrainDataFile = "data/ydotTrainData.csv"
    xTestDataFile = "data/xTestData.csv"
    yTestDataFile = "data/yTestData.csv"
    ydotTestDataFile = "data/ydotTestData.csv"
    with open(xTrainDataFile, "ab") as f:
        np.savetxt(f, X_train, delimiter=",", newline="\n")
    with open(yTrainDataFile, "ab") as f:
        np.savetxt(f, y_train, delimiter=",", newline="\n")
    with open(ydotTrainDataFile, "ab") as f:
        np.savetxt(f, ydot_train, delimiter=",", newline="\n")
    with open(xTestDataFile, "ab") as f:
        np.savetxt(f, X_test, delimiter=",", newline="\n")
    with open(yTestDataFile, "ab") as f:
        np.savetxt(f, y_test, delimiter=",", newline="\n")
    with open(ydotTestDataFile, "ab") as f:
        np.savetxt(f, ydot_test, delimiter=",", newline="\n")


def make_checkpoint_dir():
    # 1. Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 2. Create directory
    base_dir = "./checkpoints"
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


# ---- train_mlp with a closed‑over, jitted update step ----
def train_mlp(
    X_np: np.ndarray,  # shape (N, d)
    y_np: np.ndarray,  # shape (N,)
    y_dot_np: np.ndarray,  # shape (N, d)
    X_test: np.ndarray,  # shape (N_test, d)
    y_test: np.ndarray,  # shape (N_test,)
    y_dot_test: np.ndarray,  # shape (N_test, d)
    init_params=None,
    # num_blocks=6,
    # features=15,
    # hidden=64,
    hidden_sizes=(64, 64),
    lr=1e-3,
    epochs=1000,
    batch_size=512,
    seed=0,
):
    # 1) convert data to JAX
    X = jnp.array(X_np)
    y = jnp.array(y_np)
    y_dot = jnp.array(y_dot_np)

    # 2) init model + optimizer
    width = 128
    hidden_sizes = (width, width, width, width, width, width, width, width)
    hidden_sizes = (512, 512, 256, 256, 128)
    model = mlp.SimpleMLP(hidden_sizes=hidden_sizes)
    # model = mlp.PEZResidualMLP(feat_dim=X.shape[1], hidden_dim=128, n_blocks=4)
    key = random.PRNGKey(seed)
    if init_params is not None:
        params = init_params
    else:
        params = model.init(key, X[:1])  # initialize with dummy input
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    ckpt_dir = make_checkpoint_dir()
    model.save_model(ckpt_dir)
    save_every = 30

    # 3) define a jitted update step that *closes over* model & optimizer
    @jit
    def update_step(params, opt_state, X_batch, y_batch, ydot_batch, epoch_num):
        """
        One optimization step that fits both values and (non‑NaN) derivatives.

        Args:
        params      : PyTree of model parameters
        opt_state   : Optax optimizer state
        X_batch     : jnp.ndarray, shape (B, D)
        y_batch     : jnp.ndarray, shape (B,)
        ydot_batch  : jnp.ndarray, shape (B, D), may contain NaNs

        Returns:
        new_params, new_opt_state, loss_scalar
        """

        maxlam = 0.00
        lam = epoch_num / epochs * maxlam
        # lam = jnp.clip(epoch_num / epochs, 0, maxlam)

        def loss_fn(p):
            # 1) Value prediction + MSE loss
            preds = model.apply(p, X_batch)  # (B,)
            L_val = jnp.mean((preds - y_batch) ** 2)

            def single_grad(x):
                # returns shape (D,)
                return grad(lambda x0: model.apply(p, x0[None, :])[0])(x)

            dy_pred = vmap(single_grad, in_axes=0)(X_batch)  # (B, D)
            # # create penalty for negative values
            #
            mu_grads = jnp.stack(
                [
                    dy_pred[:, 4],  # turn radius
                    dy_pred[:, 6],  # range
                    dy_pred[:, 8],  # speed
                ],
                axis=1,
            )
            mu_grads_true = jnp.stack(
                [
                    ydot_batch[:, 3],  # turn radius
                    ydot_batch[:, 4],  # range
                    ydot_batch[:, 5],  # speed
                ]
            ).T
            grad_squared_error = jnp.mean((mu_grads - mu_grads_true) ** 2)
            return L_val + lam * grad_squared_error
            # value_and_grad to get both scalar loss and parameter gradients

        loss, grads = value_and_grad(loss_fn)(params)

        # optimizer update
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # 4) training loop
    n = X.shape[0]
    steps = max(1, n // batch_size)
    loss_history = []
    test_losses = []
    for epoch in tqdm.tqdm(range(epochs)):
        if epoch % save_every == 0 or epoch == epochs:
            fname = f"params_epoch{epoch:04d}.msgpack"
            path = os.path.join(ckpt_dir, fname)
            with open(path, "wb") as f:
                f.write(serialization.to_bytes(params))
        # simple shuffling
        perm = np.random.permutation(n)
        total_loss = 0.0
        for i in range(steps):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            Xb, yb, ydotb = (
                X[idx],
                y[idx],
                y_dot[idx],
            )
            params, opt_state, loss = update_step(
                params, opt_state, Xb, yb, ydotb, epoch
            )
            total_loss += loss
        loss_history.append(total_loss / steps)
        # compute test loss at epoch end
        #
        preds_test = model.apply(params, X_test)
        test_loss = jnp.mean((preds_test - y_test) ** 2)
        test_losses.append(float(test_loss))

    # save final model
    fname = "final.msgpack"
    path = os.path.join(ckpt_dir, fname)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))

    model.save_final_loss(ckpt_dir, test_losses[-1], loss_history[-1])
    return model, params, loss_history, test_losses, ckpt_dir


# ---- 4) evaluation over pursuer×evader grid ----


def evaluate_mlp_grid(
    model,
    params,
    pursuer_np: np.ndarray,  # shape (P, d1)
    evader_np: np.ndarray,  # shape (E, d2)
) -> jnp.ndarray:
    """
    Returns Z_matrix of shape (P, E), where
      Z_matrix[p,i] = model([pursuer[p], evader[i]]).
    """

    # convert to JAX
    pursuer = jnp.array(pursuer_np)
    evader = jnp.array(evader_np)

    @jit
    def eval_pair(p, e):
        x = jnp.concatenate([p, e])[None, :]  # (1, d1+d2)
        return model.apply(params, x)[0]  # scalar

    # map over evaders → returns (E,) for each p
    eval_e = jit(vmap(eval_pair, in_axes=(None, 0)))
    # then map that over all pursuers -> (P, E)
    eval_pe = jit(vmap(eval_e, in_axes=(0, None)))

    return eval_pe(pursuer, evader)


def stacked_cov(
    pursuerPositionCov,
    pursuerHeadingVar,
    pursuerSpeedVar,
    minimumTurnRadiusVar,
    pursuerRangeVar,
):
    heading_block = jnp.array([[pursuerHeadingVar]])
    speed_block = jnp.array([[pursuerSpeedVar]])
    radius_block = jnp.array([[minimumTurnRadiusVar]])
    range_block = jnp.array([[pursuerRangeVar]])

    # Assemble full covariance using jnp.block (block matrix layout)
    full_cov = jnp.block(
        [
            [
                pursuerPositionCov,
                jnp.zeros((2, 1)),
                jnp.zeros((2, 1)),
                jnp.zeros((2, 1)),
                jnp.zeros((2, 1)),
            ],
            [
                jnp.zeros((1, 2)),
                heading_block,
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
            ],
            [
                jnp.zeros((1, 2)),
                jnp.zeros((1, 1)),
                speed_block,
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
            ],
            [
                jnp.zeros((1, 2)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
                radius_block,
                jnp.zeros((1, 1)),
            ],
            [
                jnp.zeros((1, 2)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
                range_block,
            ],
        ]
    )
    return full_cov


# @jax.jit
def evaluate_surrogate_grid(model, params, pursuer, evader):
    """
    Evaluate a Flax/JAX surrogate model over all combinations of pursuer and evader parameters.

    Args:
      model:       A Flax Module (e.g., your trained MLP or ResNet).
      params:      PyTree of model parameters (restored via flax.serialization).
      pursuer_params: array-like of shape (P, d1) — P pursuer parameter vectors.
      evader_params:  array-like of shape (E, d2) — E evader parameter vectors.

    Returns:
      Z: jnp.ndarray of shape (P, E) where
          Z[p, i] = model.apply(params, concat(pursuer_params[p], evader_params[i])).
    """

    @jit
    def eval_pair(p, e):
        # p: (d1,), e: (d2,)
        x = jnp.concatenate([p, e])[None, :]  # shape (1, d1+d2)
        y = model.apply(params, x)  # shape (1,)
        return y[0]  # scalar

    # vectorize over evaders for fixed pursuer → shape (E,)
    eval_for_p = jit(vmap(eval_pair, in_axes=(None, 0)))

    # vectorize over pursuers → shape (P, E)
    eval_grid = jit(vmap(eval_for_p, in_axes=(0, None)))

    # compute full matrix
    return eval_grid(pursuer, evader)


@jax.jit
def create_pursuer_params(
    pursuerPositionCov,
    pursuerHeadingVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    pursuerSpeed,
    pursuerSpeedVar,
):
    pursuerParams = jnp.array(
        [
            [
                pursuerPositionCov[0, 0],
                pursuerPositionCov[1, 1],
                pursuerPositionCov[0, 1],
                pursuerHeadingVar,
                minimumTurnRadius,
                minimumTurnRadiusVar,
                pursuerRange,
                pursuerRangeVar,
                pursuerSpeed,
                pursuerSpeedVar,
            ]
        ]
    )
    return pursuerParams


def unstack_pursuer_params(pursuerParams):
    pursuerParams = pursuerParams[0]
    pursuerPositionCov = jnp.array(
        [[pursuerParams[0], pursuerParams[2]], [pursuerParams[2], pursuerParams[1]]]
    )
    pursuerHeading = pursuerParams[3]
    pursuerHeadingVar = pursuerParams[4]
    minimumTurnRadius = pursuerParams[5]
    minimumTurnRadiusVar = pursuerParams[6]
    pursuerRange = pursuerParams[7]
    pursuerRangeVar = pursuerParams[8]
    pursuerSpeed = pursuerParams[9]
    pursuerSpeedVar = pursuerParams[10]
    return (
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
    )


def plot_results(evaderHeading, evaderSpeed, model, restored_params, pursuerParams):
    rangeX = 1.5
    numPoints = 100
    x = jnp.linspace(-rangeX, rangeX, numPoints)
    y = jnp.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    evaderHeadings = np.ones_like(X) * evaderHeading
    evaderPositions = np.column_stack((X, Y))
    evaderSpeeds = np.ones_like(X) * evaderSpeed
    evaderParams = np.hstack(
        [evaderPositions, evaderHeadings[:, None], evaderSpeeds[:, None]]
    )
    Z = evaluate_surrogate_grid(model, restored_params, pursuerParams, evaderParams)

    Z = Z.reshape((numPoints, numPoints))

    X = X.reshape((numPoints, numPoints))
    Y = Y.reshape((numPoints, numPoints))
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_title("Nurearl Network PEZ")
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    plt.clabel(c, inline=True, fontsize=20)
    # ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
    pursuerPosition = np.array([0.0, 0.0])
    (
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
    ) = unstack_pursuer_params(pursuerParams)

    fig2, ax2 = plt.subplots()
    ax2.set_aspect("equal")
    ax2.set_title("Monte Carlo PEZ")
    ZTrue, _, _, _, _, _, _ = dubinsPEZ.mc_dubins_PEZ(
        jnp.array([X.flatten(), Y.flatten()]).T,
        evaderHeadings,
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        0.0,
    )
    ZTrue = ZTrue.reshape((numPoints, numPoints))
    X = X.reshape((numPoints, numPoints))
    Y = Y.reshape((numPoints, numPoints))
    c = ax2.contour(
        X,
        Y,
        ZTrue,
        levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    plt.clabel(c, inline=True, fontsize=20)
    fig3, ax3 = plt.subplots()
    ax3.set_aspect("equal")
    ZMC = ZTrue
    rmse = jnp.sqrt(jnp.mean((Z - ZTrue) ** 2))
    average_abs_diff = jnp.mean(jnp.abs(Z - ZMC))
    max_abs_diff = jnp.max(jnp.abs(Z - ZMC))
    ZMC = ZMC.reshape(numPoints, numPoints)
    ZTrue = ZTrue.reshape(numPoints, numPoints)
    # write rmse on image
    ax3.text(
        0.0,
        0.9,
        f"RMSE: {rmse:.4f}",
        fontsize=12,
        ha="center",
        va="center",
        color="white",
    )
    ax3.text(
        0.0,
        0.7,
        f"Avg Abs Diff: {average_abs_diff:.4f}",
        fontsize=12,
        ha="center",
        va="center",
        color="white",
    )
    ax3.text(
        0.0,
        0.5,
        f"Max Abs Diff: {max_abs_diff:.4f}",
        fontsize=12,
        ha="center",
        va="center",
        color="white",
    )

    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    c = ax3.pcolormesh(X, Y, jnp.abs(Z - ZMC))
    # make colorbar smaller
    cb = plt.colorbar(c, ax=ax3, shrink=0.5)
    plt.show()


def load_model(folder, net):
    file = os.path.join(folder, "final.msgpack")
    with open(file, "rb") as f:
        byte_data = f.read()
    if net == "mlp":
        model = mlp.SimpleMLP()
        model.load_model(folder)
    else:
        model = mlp.PEZResidualMLP(feat_dim=15, hidden_dim=64, n_blocks=4)
        model.load_model(folder)

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 15))  # D = your input dimension
    template_params = model.init(rng, dummy_input)
    restored_params = serialization.from_bytes(template_params, byte_data)
    return model, restored_params


start = time.time()
saveDir = "./checkpoints/20250429_151150/"
# saveDir = "./checkpoints/20250425_170039/"
model, restored_params = load_model(saveDir, "mlp")
print("load model time", time.time() - start)


@jax.jit
def nueral_network_pez(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    evaderPositions -= pursuerPosition
    evaderHeadings -= pursuerHeading
    rotation = jnp.array(
        [
            [jnp.cos(pursuerHeading), -jnp.sin(pursuerHeading)],
            [jnp.sin(pursuerHeading), jnp.cos(pursuerHeading)],
        ]
    )
    evaderPositions = evaderPositions @ rotation
    pursuerPositionCov = rotation.T @ pursuerPositionCov @ rotation

    pursuerParams = create_pursuer_params(
        pursuerPositionCov,
        pursuerHeadingVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
    )
    evaderSpeeds = jnp.ones_like(evaderPositions[:, 0]) * evaderSpeed
    evaderParams = jnp.hstack(
        [evaderPositions, evaderHeadings[:, None], evaderSpeeds[:, None]]
    )
    Z = evaluate_surrogate_grid(model, restored_params, pursuerParams, evaderParams)
    return Z.squeeze(), 0, 0


def linear_pez_from_x_single(X):
    pursuerPosition = jnp.array([0.0, 0.0])
    pursuerPositionCov = jnp.array([[X[0], X[2]], [X[2], X[1]]]) + jnp.eye(2) * 1e-6
    pursuerHeadingVar = X[3]
    minimumTurnRadius = X[4]
    minimumTurnRadiusVar = X[5]
    pursuerRange = X[6]
    pursuerRangeVar = X[7]
    pursuerSpeed = X[8]
    pursuerSpeedVar = X[9]
    evaderPosition = X[10:12]
    evaderHeading = X[12]
    evaderSpeed = X[13]
    pursuerHeading = 0.0
    z, _, _ = dubinsPEZ.linear_dubins_PEZ_single(
        evaderPosition,
        evaderHeading,
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        0.0,
    )
    combined_covariance = stacked_cov(
        pursuerPositionCov,
        pursuerHeadingVar,
        pursuerSpeedVar,
        minimumTurnRadiusVar,
        pursuerRangeVar,
    )
    det = jnp.linalg.det(combined_covariance)
    trace = jnp.trace(combined_covariance)
    return z.squeeze(), det, trace


linear_pez_from_x = jax.jit(jax.vmap(linear_pez_from_x_single, in_axes=(0,)))


def nueral_network_pez_from_x_single(X):
    pursuerPosition = jnp.array([0.0, 0.0])
    pursuerPositionCov = jnp.array([[X[0], X[2]], [X[2], X[1]]]) + jnp.eye(2) * 1e-6
    pursuerHeadingVar = X[3]
    minimumTurnRadius = X[4]
    minimumTurnRadiusVar = X[5]
    pursuerRange = X[6]
    pursuerRangeVar = X[7]
    pursuerSpeed = X[8]
    pursuerSpeedVar = X[9]
    evaderPosition = X[10:12]
    evaderHeading = X[12]
    evaderSpeed = X[13]
    pursuerHeading = 0.0
    z, _, _ = dubinsPEZ.dubins_pez_numerical_integration_sparse(
        jnp.array([evaderPosition]),
        jnp.array([evaderHeading]),
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        0.0,
        dubinsPEZ.nodes,
        dubinsPEZ.weights,
    )
    return z.squeeze()


numerical_pez_from_x = jax.jit(jax.vmap(nueral_network_pez_from_x_single, in_axes=(0,)))


def batch_numerical_pez_from_x(X):
    batch_size = 200
    y = []
    for i in tqdm.tqdm(range(0, X.shape[0], batch_size)):
        j = i + batch_size
        if j > X.shape[0]:
            j = X.shape[0]
        ytemp = numerical_pez_from_x(X[i:j])
        y.append(ytemp)
    return np.concatenate(y, axis=0)


def binning(abs_error, trace):
    # binning
    #
    bins = np.linspace(0, 1.5, 30)
    bin_means = []
    for i in range(len(bins) - 1):
        bin_mask = (trace >= bins[i]) & (trace < bins[i + 1])
        bin_mean = jnp.mean(abs_error[bin_mask])
        # bin_mean = jnp.max(abs_error[bin_mask])
        print(f"Bin {i}: {bins[i]} - {bins[i + 1]}: {bin_mean}")
        bin_means.append(bin_mean)
    return bins, bin_means


def compute_loss_linear_pez(
    X_test,
    y_test,
):
    saveDir = "./checkpoints/20250429_151150/"
    model, restored_params = load_model(saveDir, "mlp")
    y_pred_nn = model.apply(restored_params, X_test)
    y_pred_lin, det, trace = linear_pez_from_x(X_test)

    # y_pred_numerical = batch_numerical_pez_from_x(X_test)

    loss_lin = jnp.mean((y_pred_lin - y_test) ** 2)
    loss_nn = jnp.mean((y_pred_nn - y_test) ** 2)
    # loss_numerical = jnp.mean((y_pred_numerical - y_test) ** 2)
    print("linear loss", loss_lin)
    print("nn loss", loss_nn)
    # print("numerical loss", loss_numerical)
    print("linear shape", y_pred_lin.shape)
    print("nn shape", y_pred_nn.shape)
    average_abs_diff_lin = jnp.mean(jnp.abs(y_pred_lin - y_test))
    average_abs_diff_nn = jnp.mean(jnp.abs(y_pred_nn - y_test))
    # average_abs_diff_numerical = jnp.mean(jnp.abs(y_pred_numerical - y_test))
    print("average abs diff linear", average_abs_diff_lin)
    print("average abs diff nn", average_abs_diff_nn)
    # print("average abs diff numerical", average_abs_diff_numerical)

    abs_error_lin = jnp.abs(y_pred_lin - y_test)
    abs_error_nn = jnp.abs(y_pred_nn - y_test)
    # abs_error_numerical = jnp.abs(y_pred_numerical - y_test)

    # sort by trace
    tracesortIndex = jnp.argsort(trace)
    trace = trace[tracesortIndex]
    abs_error_lin = abs_error_lin[tracesortIndex]
    abs_error_nn = abs_error_nn[tracesortIndex]
    # abs_error_numerical = abs_error_numerical[tracesortIndex]

    bins, bin_means_lin = binning(abs_error_lin, trace)
    bins, bin_means_nn = binning(abs_error_nn, trace)
    # bins, bin_means_numerical = binning(abs_error_numerical, trace)

    fig2, ax2 = plt.subplots()
    ax2.plot(bins[:-1], bin_means_lin, label="linear PEZ")
    ax2.plot(bins[:-1], bin_means_nn, label="NN PEZ")
    # ax2.plot(bins[:-1], bin_means_numerical, label="numerical PEZ")
    ax2.set_title("Average Absolute Error")
    ax2.set_xlabel(r"Trace of $\Sigma_{\Theta_P}$")
    ax2.set_ylabel("Average Absolute Error")
    ax2.legend()

    plt.show()


def main():
    generateData = False
    if generateData:
        rng_args = (
            0.5,  # pursuerPositionCovMax
            0.5,  # pursuerHeadingVarMax
            (0.1, 1.5),  # pursuerTurnRadiusRange
            0.025,  # pursuerTurnRadiusVarMax
            (1.0, 3.0),  # pursuerRangeRange
            0.5,  # pursuerRangeVarMax
            (0.5, 3),  # pursuerSpeedRange
            0.5,  # pursuerSpeedVarMax
            (-3, 3),  # evaderXPositionRange
            (-3, 3),  # evaderYPositionRange
            (-np.pi, np.pi),  # evaderHeadingRange
            (0.5, 2),  # evaderSpeedRange
        )
        numberOfSamples = 10000000
        create_input_output_data_full(numberOfSamples, *rng_args)
        # generate_data where all but heading var is 0
        # rng_args = (
        #     0.0,  # pursuerPositionCovMax
        #     0.5,  # pursuerHeadingVarMax
        #     (0.1, 1.5),  # pursuerTurnRadiusRange
        #     0.0,  # pursuerTurnRadiusVarMax
        #     (1.0, 3.0),  # pursuerRangeRange
        #     0.0,  # pursuerRangeVarMax
        #     (0.5, 3),  # pursuerSpeedRange
        #     0.0,  # pursuerSpeedVarMax
        #     (-3, 3),  # evaderXPositionRange
        #     (-3, 3),  # evaderYPositionRange
        #     (-np.pi, np.pi),  # evaderHeadingRange
        #     (0.5, 2),  # evaderSpeedRange
        # )
        # numberOfSamples = 1
        # create_input_output_data_full(numberOfSamples, *rng_args)

    trainModel = False
    if trainModel:
        X_test = np.genfromtxt("data/xTestData.csv", delimiter=",")
        y_test = np.genfromtxt("data/yTestData.csv", delimiter=",")
        ydot_test = np.genfromtxt("data/ydotTestData.csv", delimiter=",")
        X_train = np.genfromtxt("data/xTrainData.csv", delimiter=",")
        y_train = np.genfromtxt("data/yTrainData.csv", delimiter=",")
        ydot_train = np.genfromtxt("data/ydotTrainData.csv", delimiter=",")
        print("y number of nan", np.sum(np.isnan(y_train)))
        X_train = X_train[~np.isnan(y_train)]
        y_train = y_train[~np.isnan(y_train)]

        model, params, loss, test_loss, saveDir = train_mlp(
            X_train,
            y_train,
            ydot_train,
            X_test,
            y_test,
            ydot_test,
            epochs=300,
            batch_size=1024,
        )
        fig, ax = plt.subplots()
        ax.plot(loss)
        ax.set_title("Training Loss")
        fig2, ax2 = plt.subplots()
        ax2.plot(test_loss)
        ax2.set_title("Test Loss")
        print("fine loss", loss[-1])
        print("fine test loss", test_loss[-1])

    plotTest = False
    if plotTest:
        pursuerPosition = np.array([0.0, 0.0])
        # pursuerPositionCov = np.array([[0.025, -0.04], [-0.04, 0.1]])
        pursuerPositionCov = np.array([[0.025, -0.04], [-0.04, 0.1]])
        # pursuerPositionCov = np.array([[0.000000000001, 0.0], [0.0, 0.00000000001]])
        pursuerHeading = (0.0 / 4.0) * np.pi
        pursuerHeadingVar = 0.0

        pursuerSpeed = 2.0
        pursuerSpeedVar = 0.1
        # pursuerSpeedVar = 0.0
        pursuerRange = 1.0

        pursuerRangeVar = 0.3
        # pursuerRangeVar = 0.0

        minimumTurnRadius = 0.2
        minimumTurnRadiusVar = 0.01
        # minimumTurnRadiusVar = 0.0

        captureRadius = 0.0

        evaderHeading = jnp.array([(10.0 / 20.0) * np.pi])
        # evaderHeading = jnp.array((0.0 / 20.0) * np.pi)

        evaderSpeed = 0.5
        evaderPosition = np.array([[-0.25, 0.35]])
        pursuerParams = create_pursuer_params(
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadingVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            pursuerSpeed,
            pursuerSpeedVar,
        )
        # saveDir = "./checkpoints/20250423_171138/"
        loadDir = saveDir
        model, restored_params = load_model(loadDir, "mlp")
        #
        plot_results(
            evaderHeading,
            evaderSpeed,
            model,
            restored_params,
            pursuerParams,
        )
    compareLoss = True
    if compareLoss:
        X_test = np.genfromtxt("data/xTestData.csv", delimiter=",")
        y_test = np.genfromtxt("data/yTestData.csv", delimiter=",")

        compute_loss_linear_pez(X_test, y_test)


if __name__ == "__main__":
    main()
    plt.show()
