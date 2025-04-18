import enum
import jax
import tqdm
import os

import optax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jax import random, jit, vmap, grad
from flax import linen as nn
from flax import serialization
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

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

mc_dubins_pez_vmap = jax.jit(
    jax.vmap(
        dubinsPEZ.mc_dubins_PEZ,
        in_axes=(
            None,
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            None,
        ),
    )
)


# evaderPositions,
# evaderHeadings,
# evaderSpeed,
# pursuerPosition,
# pursuerPositionCov,
# pursuerHeading,
# pursuerHeadingVar,
# pursuerSpeed,
# pursuerSpeedVar,
# minimumTurnRadius,
# minimumTurnRadiusVar,
# pursuerRange,
# pursuerRangeVar,
# captureRadius,
def create_covariance_matrix_single(var1, var2, cov):
    return jnp.array([[var1, cov], [cov, var2]])


create_covariance_matrix = jax.jit(
    jax.vmap(create_covariance_matrix_single, in_axes=(0, 0, 0))
)


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


# ---- your Flax model ----
class ResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        # skip connection
        x = x + residual
        return nn.relu(x)


class ResMLP(nn.Module):
    hidden_sizes: list[int]

    @nn.compact
    def __call__(self, x):
        # initial “embedding” layer
        x = nn.Dense(self.hidden_sizes[0])(x)
        x = nn.relu(x)

        # stack residual blocks
        for h in self.hidden_sizes:
            x = ResBlock(h)(x)

        # final regression head
        x = nn.Dense(1)(x)
        return x


# ---- train_mlp with a closed‑over, jitted update step ----
def train_mlp(
    X_np: np.ndarray,  # shape (N, d)
    y_np: np.ndarray,  # shape (N,)
    hidden_sizes=[64, 64, 64, 64, 64],
    lr=1e-3,
    epochs=1000,
    batch_size=512,
    seed=0,
):
    # 1) convert data to JAX
    X = jnp.array(X_np)
    y = jnp.array(y_np)

    # 2) init model + optimizer
    model = ResMLP(hidden_sizes)
    key = random.PRNGKey(seed)
    params = model.init(key, X[:1])  # initialize with dummy input
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    save_every = 100

    # 3) define a jitted update step that *closes over* model & optimizer
    @jit
    def update_step(params, opt_state, X_batch, y_batch):
        def loss_fn(p):
            preds = model.apply(p, X_batch)
            return jnp.mean((preds - y_batch) ** 2)

        grads = grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # 4) training loop
    n = X.shape[0]
    steps = max(1, n // batch_size)
    checkpoint_dir = "./checkpoints/"
    # tqdm loop
    for epoch in tqdm.tqdm(range(epochs)):
        if epoch % save_every == 0 or epoch == epochs:
            fname = f"params_epoch{epoch:04d}.msgpack"
            path = os.path.join(checkpoint_dir, fname)
            with open(path, "wb") as f:
                f.write(serialization.to_bytes(params))
            print(f"[Epoch {epoch}] checkpoint saved to {path}")
        # simple shuffling
        perm = np.random.permutation(n)
        for i in range(steps):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            Xb, yb = X[idx], y[idx]
            params, opt_state = update_step(params, opt_state, Xb, yb)

    return model, params


# ---- 4) evaluation over pursuer×evader grid ----


def evaluate_mlp_grid(
    model: ResMLP,
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

    # numberOfSamples = 100000
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
    # model, params = train_mlp(X, y, epochs=5000)
    X = np.genfromtxt("data/xData.csv", delimiter=",")
    y = np.genfromtxt("data/yData.csv", delimiter=",")
    train_mlp(
        X,
        y,
        epochs=5000,
        batch_size=512,
    )


if __name__ == "__main__":
    main()
