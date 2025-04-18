import enum
import jax
import tqdm
import os

import optax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jax import random, jit, vmap, grad, value_and_grad
from flax import linen as nn
from flax import serialization
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import qmc


import dubinsEZ
import dubinsPEZ

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
    ) = unstack_pursuer_params(X)
    evaderPosition = X[11:13]
    evaderHeading = X[13]
    evaderSpeed = X[14]
    return dubinsPEZ.mc_dubins_PEZ_Single(
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


mc_combined_input_jac_single = jax.jit(jax.grad(mc_combined_input_single))
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
    pursuerHeadingRange: tuple[float, float],
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
    seed: int = 0,
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
    d = 15  # total input dims
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
    phead = m1(unit[:, col], *pursuerHeadingRange)
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
            phead,
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
    epos,
    ehead,
    espd,
    ppos,
    ppos_cov,
    phead,
    phvar,
    pspd,
    psvar,
    ptrad,
    ptvar,
    prng,
    prvar,
    other_arg=0.0,
    batch_size=200,
):
    """
    epos:   (N,2)
    ehead:  (N,)
    espd:   (N,)
    ppos:   (N,2)
    ppos_cov:(N,2,2)
    phead,phvar,pspd,psvar,ptrad,ptvar,prng,prvar: all (N,)
    other_arg: final constant argument for mc_dubins_pez_vmap
    """
    N = epos.shape[0]
    ys = []
    ydot = []
    for i in range(0, N, batch_size):
        j = i + batch_size
        yb, *_ = mc_dubins_pez_vmap(
            epos[i:j],
            ehead[i:j],
            espd[i:j],
            ppos[i:j],
            ppos_cov[i:j],
            phead[i:j],
            phvar[i:j],
            pspd[i:j],
            psvar[i:j],
            ptrad[i:j],
            ptvar[i:j],
            prng[i:j],
            prvar[i:j],
            other_arg,
        )
        ys.append(yb)

    return np.concatenate(ys, axis=0)  # shape (N,)


# -----------------------------------------------------------------------------
# 3) End‑to‑end data creation
# -----------------------------------------------------------------------------
def create_input_output_data_full(
    n_samples: int,
    # same args as sample_inputs_lhs...
    *ranges_and_maxes,
    seed: int = 0,
):
    # unpack all the args you passed through; for brevity assume same order
    inp = sample_inputs_lhs(n_samples, *ranges_and_maxes, seed=seed)

    # pursuer positions are zero as before
    ppos_zero = np.zeros((n_samples, 2))

    # run batched MC‑PEZ
    y = batched_mc_dubins_pez(
        inp["epos"],
        inp["ehead"],
        inp["espd"],
        ppos_zero,
        inp["ppos_cov"],
        inp["phead"],
        inp["phvar"],
        inp["pspd"],
        inp["psvar"],
        inp["ptrad"],
        inp["ptvar"],
        inp["prng"],
        inp["prvar"],
        other_arg=0.0,
        batch_size=200,
    )

    # assemble X matrix in the same column order
    X = np.column_stack(
        [
            inp["pvx"],
            inp["pvy"],
            inp["pcov"],
            inp["phead"],
            inp["phvar"],
            inp["ptrad"],
            inp["ptvar"],
            inp["prng"],
            inp["prvar"],
            inp["pspd"],
            inp["psvar"],
            inp["epos"][:, 0],
            inp["epos"][:, 1],
            inp["ehead"],
            inp["espd"],
        ]
    )
    # save data to csv
    xDataFile = "data/xData.csv"
    yDataFile = "data/yData.csv"
    with open(xDataFile, "ab") as f:
        np.savetxt(f, X, delimiter=",", newline="\n")
    with open(yDataFile, "ab") as f:
        np.savetxt(f, y, delimiter=",", newline="\n")

    return X, y


def create_input_output_data(
    n_samples,
    pursuerPositionCovMax,
    pursuerHeadingRange,
    pursuerHeadingVarMax,
    pursuerTurnRadiusRange,
    pursuerTurnRadiusVarMax,
    pursuerRangeRange,
    pursuerRangeVarMax,
    pursuerSpeedRange,
    pursuerSpeedVarMax,
    evaderXPositionRange,
    evaderYPositionRange,
    evaderHeadingRange,
    evaderSpeedRange,
):
    pursuerPositions = np.zeros((n_samples, 2))
    pursuerVarX = np.random.uniform(0, pursuerPositionCovMax, n_samples)
    pursuerVarY = np.random.uniform(0, pursuerPositionCovMax, n_samples)
    pursuerXYcov = np.random.uniform(
        -np.sqrt(pursuerVarX * pursuerVarY),
        np.sqrt(pursuerVarX * pursuerVarY),
        n_samples,
    )
    pursuerPositionCov = create_covariance_matrix(
        pursuerVarX, pursuerVarY, pursuerXYcov
    )
    pursuerHeading = np.random.uniform(
        pursuerHeadingRange[0], pursuerHeadingRange[1], n_samples
    )
    pursuerHeadingVar = np.random.uniform(0, pursuerHeadingVarMax, n_samples)
    pursuerTurnRadius = np.random.uniform(
        pursuerTurnRadiusRange[0], pursuerTurnRadiusRange[1], n_samples
    )
    purserTrunRadiusVar = np.random.uniform(0, pursuerTurnRadiusVarMax, n_samples)
    pursuerRange = np.random.uniform(
        pursuerRangeRange[0], pursuerRangeRange[1], n_samples
    )
    pursuerRangeVar = np.random.uniform(0, pursuerRangeVarMax, n_samples)
    pursuerSpeed = np.random.uniform(
        pursuerSpeedRange[0], pursuerSpeedRange[1], n_samples
    )
    pursuerSpeedVar = np.random.uniform(0, pursuerSpeedVarMax, n_samples)
    evaderXPosition = np.random.uniform(
        evaderXPositionRange[0], evaderXPositionRange[1], n_samples
    )
    evaderYPosition = np.random.uniform(
        evaderYPositionRange[0], evaderYPositionRange[1], n_samples
    )
    evaderHeadings = np.random.uniform(
        evaderHeadingRange[0], evaderHeadingRange[1], n_samples
    )
    evaderPositions = np.stack((evaderXPosition, evaderYPosition), axis=1)
    evaderSpeed = np.random.uniform(evaderSpeedRange[0], evaderSpeedRange[1], n_samples)
    y, _, _, _, _, _, _ = mc_dubins_pez_vmap(
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
        pursuerPositions,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        pursuerTurnRadius,
        purserTrunRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        0.0,
    )

    X = np.hstack(
        [
            pursuerVarX[:, None],
            pursuerVarY[:, None],
            pursuerXYcov[:, None],
            pursuerHeading[:, None],
            pursuerHeadingVar[:, None],
            pursuerTurnRadius[:, None],
            purserTrunRadiusVar[:, None],
            pursuerRange[:, None],
            pursuerRangeVar[:, None],
            pursuerSpeed[:, None],
            pursuerSpeedVar[:, None],
            evaderXPosition[:, None],
            evaderYPosition[:, None],
            evaderHeadings[:, None],
            evaderSpeed[:, None],
        ]
    )
    return X, y


def generate_data(
    n_samples,
    pursuerPositionCovMax,
    pursuerHeadingRange,
    pursuerHeadingVarMax,
    pursuerTurnRadiusRange,
    pursuerTurnRadiusVarMax,
    pursuerRangeRange,
    pursuerRangeVarMax,
    pursuerSpeedRange,
    pursuerSpeedVarMax,
    evaderXPositionRange,
    evaderYPositionRange,
    evaderHeadingRange,
    evaderSpeedRange,
):
    maxSamplesOnGpu = 200
    numRounds = n_samples // maxSamplesOnGpu
    xDataFile = "data/xData.csv"
    yDataFile = "data/yData.csv"
    for i in tqdm.trange(numRounds):
        X, y = create_input_output_data(
            maxSamplesOnGpu,
            pursuerPositionCovMax,
            pursuerHeadingRange,
            pursuerHeadingVarMax,
            pursuerTurnRadiusRange,
            pursuerTurnRadiusVarMax,
            pursuerRangeRange,
            pursuerRangeVarMax,
            pursuerSpeedRange,
            pursuerSpeedVarMax,
            evaderXPositionRange,
            evaderYPositionRange,
            evaderHeadingRange,
            evaderSpeedRange,
        )
        with open(xDataFile, "ab") as f:
            np.savetxt(f, X, delimiter=",", newline="\n")
        with open(yDataFile, "ab") as f:
            np.savetxt(f, y, delimiter=",", newline="\n")


# class ResBlock(nn.Module):
#     features: int
#     hidden: int
#
#     @nn.compact
#     def __call__(self, x):
#         residual = x
#         x = nn.LayerNorm()(x)
#         x = nn.Dense(self.hidden)(x)
#         x = nn.relu(x)
#         x = nn.Dense(self.features)(x)
#         return x + residual
#
#
# class MLP(nn.Module):
#     num_blocks: int
#     features: int
#     hidden: int
#
#     @nn.compact
#     def __call__(self, x):
#         # x: (batch, d_in)
#         x = nn.Dense(self.features)(x)
#         for _ in range(self.num_blocks):
#             x = ResBlock(self.features, self.hidden)(x)
#         x = nn.LayerNorm()(x)
#         x = nn.tanh(x)
#         x = nn.Dense(1)(x)
#         return nn.sigmoid(x).squeeze(-1)
#


class SimpleMLP(nn.Module):
    hidden_sizes: list[int] = (64, 64)  # two hidden layers of 64 units

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, D)
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)  # dense → hidden dimension
            # x = nn.relu(x)  # activation
            x = nn.tanh(x)  # activation
        x = nn.Dense(1)(x)  # final linear layer → scalar
        return nn.sigmoid(x).squeeze(-1)


# ---- train_mlp with a closed‑over, jitted update step ----
def train_mlp(
    X_np: np.ndarray,  # shape (N, d)
    y_np: np.ndarray,  # shape (N,)
    X_test: np.ndarray,  # shape (N_test, d)
    y_test: np.ndarray,  # shape (N_test,)
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

    # 2) init model + optimizer
    # model = ResMLP(num_blocks=num_blocks, features=features, hidden=hidden)
    model = SimpleMLP(hidden_sizes=hidden_sizes)
    key = random.PRNGKey(seed)
    if init_params is not None:
        params = init_params
    else:
        params = model.init(key, X[:1])  # initialize with dummy input
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    save_every = 1000

    # 3) define a jitted update step that *closes over* model & optimizer
    @jit
    def update_step(params, opt_state, X_batch, y_batch):
        def loss_fn(p):
            preds = model.apply(p, X_batch)
            return jnp.mean((preds - y_batch) ** 2)

        loss, grads = value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # 4) training loop
    n = X.shape[0]
    steps = max(1, n // batch_size)
    checkpoint_dir = "./checkpoints/"
    loss_history = []
    test_losses = []
    for epoch in tqdm.tqdm(range(epochs)):
        if epoch % save_every == 0 or epoch == epochs:
            fname = f"params_epoch{epoch:04d}.msgpack"
            path = os.path.join(checkpoint_dir, fname)
            with open(path, "wb") as f:
                f.write(serialization.to_bytes(params))
        # simple shuffling
        perm = np.random.permutation(n)
        total_loss = 0.0
        for i in range(steps):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            Xb, yb = X[idx], y[idx]
            params, opt_state, loss = update_step(params, opt_state, Xb, yb)
            total_loss += loss
        loss_history.append(total_loss / steps)
        # compute test loss at epoch end
        #
        preds_test = model.apply(params, X_test)
        test_loss = jnp.mean((preds_test - y_test) ** 2)
        test_losses.append(float(test_loss))

    # save final model
    fname = "final.msgpack"
    path = os.path.join(checkpoint_dir, fname)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))

    return model, params, loss_history, test_losses


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
    heading_block = np.array([[pursuerHeadingVar]])
    speed_block = np.array([[pursuerSpeedVar]])
    radius_block = np.array([[minimumTurnRadiusVar]])
    range_block = np.array([[pursuerRangeVar]])

    # Assemble full covariance using np.block (block matrix layout)
    full_cov = np.block(
        [
            [
                pursuerPositionCov,
                np.zeros((2, 1)),
                np.zeros((2, 1)),
                np.zeros((2, 1)),
                np.zeros((2, 1)),
            ],
            [
                np.zeros((1, 2)),
                heading_block,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
            ],
            [
                np.zeros((1, 2)),
                np.zeros((1, 1)),
                speed_block,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
            ],
            [
                np.zeros((1, 2)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                radius_block,
                np.zeros((1, 1)),
            ],
            [
                np.zeros((1, 2)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                range_block,
            ],
        ]
    )
    return full_cov


def evaluate_surrogate_grid(model, params, pursuer_params, evader_params):
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
    # Convert inputs to JAX arrays
    pursuer = jnp.array(pursuer_params)  # shape (P, d1)
    evader = jnp.array(evader_params)  # shape (E, d2)

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


def create_pursuer_params(
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    pursuerSpeed,
    pursuerSpeedVar,
):
    pursuerParams = np.array(
        [
            [
                pursuerPositionCov[0, 0],
                pursuerPositionCov[1, 1],
                pursuerPositionCov[0, 1],
                pursuerHeading,
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
    pursuerPositionCov = np.array(
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
    print("pursuerParams", pursuerParams.shape)
    print("evaderParams", evaderParams.shape)
    Z = evaluate_surrogate_grid(model, restored_params, pursuerParams, evaderParams)

    Z = Z.reshape((numPoints, numPoints))
    print("Z", Z)

    X = X.reshape((numPoints, numPoints))
    Y = Y.reshape((numPoints, numPoints))
    fig, ax = plt.subplots()
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


def load_model(file, hidden_sizes):
    with open(file, "rb") as f:
        byte_data = f.read()

    # model = ResMLP(
    #     hidden=64,
    #     num_blocks=6,
    #     features=15,
    # )
    model = SimpleMLP(hidden_sizes=hidden_sizes)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 15))  # D = your input dimension
    template_params = model.init(rng, dummy_input)
    restored_params = serialization.from_bytes(template_params, byte_data)
    return model, restored_params


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerPositionCovMax = 0.5

    pursuerHeadingRange = np.array([-np.pi, np.pi])
    pursuerHeadingVarMax = 0.5

    pursuerSpeedRange = np.array([0.5, 3.0])
    pursuerSpeedVarMax = 0.5

    pursuerRangeRange = np.array([1.0, 3.0])
    pursuerRangeVarMax = 0.5

    minimumTurnRadiusRange = np.array([0.1, 1.5])
    minimumTurnRadiusVarMax = 0.025
    # minimumTurnRadiusVar = 0.00000000001

    captureRadius = 0.0

    evaderHeadingRange = np.array([-np.pi, np.pi])
    evaderXPositionRange = np.array([-3.0, 3.0])
    evaderYPositionRange = np.array([-3.0, 3.0])

    evaderSpeedRange = np.array([0.5, 2.0])
    rng_args = (
        0.5,  # pursuerPositionCovMax
        (-np.pi, np.pi),  # pursuerHeadingRange
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

    generateData = True
    if generateData:
        numberOfSamples = 500000
        X, y = create_input_output_data_full(numberOfSamples, *rng_args, seed=42)

        # generate_data(
        #     numberOfSamples,
        #     pursuerPositionCovMax,
        #     pursuerHeadingRange,
        #     pursuerHeadingVarMax,
        #     minimumTurnRadiusRange,
        #     minimumTurnRadiusVarMax,
        #     pursuerRangeRange,
        #     pursuerRangeVarMax,
        #     pursuerSpeedRange,
        #     pursuerSpeedVarMax,
        #     evaderXPositionRange,
        #     evaderYPositionRange,
        #     evaderHeadingRange,
        #     evaderSpeedRange,
        # )
    numHidden = 64
    hidden_sizes = [
        numHidden,
        numHidden,
        numHidden,
        numHidden,
        numHidden,
        numHidden,
        numHidden,
        numHidden,
        numHidden,
        numHidden,
    ]
    trainModel = True
    if trainModel:
        X = np.genfromtxt("data/xData.csv", delimiter=",")
        print("X", X.shape)
        y = np.genfromtxt("data/yData.csv", delimiter=",")
        # model, restored_params = load_model("./checkpoints/params_epoch0900.msgpack")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model, params, loss, test_loss = train_mlp(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=300,
            batch_size=1024,
            hidden_sizes=hidden_sizes,
            # num_blocks=numBlocks,
            # features=features,
            # hidden=numHidden,
            # init_params=restored_params,
        )
        fig, ax = plt.subplots()
        ax.plot(loss)
        ax.set_title("Training Loss")
        fig2, ax2 = plt.subplots()
        ax2.plot(test_loss)
        ax2.set_title("Test Loss")

    pursuerPosition = np.array([0.0, 0.0])
    pursuerPositionCov = np.array([[0.025, -0.04], [-0.04, 0.1]])
    # pursuerPositionCov = np.array([[0.000000000001, 0.0], [0.0, 0.00000000001]])

    pursuerHeading = (0.0 / 4.0) * np.pi
    pursuerHeadingVar = 0.5

    pursuerSpeed = 2.0
    pursuerSpeedVar = 0.3

    pursuerRange = 1.0
    pursuerRangeVar = 0.1
    pursuerRangeVar = 0.0

    minimumTurnRadius = 0.2
    minimumTurnRadiusVar = 0.005
    # minimumTurnRadiusVar = 0.00000000001

    captureRadius = 0.0

    evaderHeading = jnp.array([(5.0 / 20.0) * np.pi])
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
    model, restored_params = load_model("./checkpoints/final.msgpack", hidden_sizes)
    #
    plot_results(
        evaderHeading,
        evaderSpeed,
        model,
        restored_params,
        pursuerParams,
    )


if __name__ == "__main__":
    main()
