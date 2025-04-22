from flax import linen as nn
import jax.numpy as jnp
import jax
import numpy as np


class SimpleMLP(nn.Module):
    hidden_sizes: list[int] = (64, 64)  # two hidden layers of 64 units

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, D)
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)  # dense → hidden dimension
            # x = nn.relu(x)  # activation
            # x = nn.tanh(x)  # activation
            x = nn.silu(x)  # activation
        x = nn.Dense(1)(x)  # final linear layer → scalar
        return nn.sigmoid(x).squeeze(-1)

    def save_model(self, dir):
        hidden_sizes = np.array(self.hidden_sizes)
        filepath = f"{dir}/mlp_hidden_sizes.txt"
        np.savetxt(filepath, hidden_sizes, delimiter=",")

    def load_model(self, dir):
        filepath = f"{dir}/mlp_hidden_sizes.txt"
        hidden_sizes = np.loadtxt(filepath, delimiter=",")
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        return hidden_sizes

    def save_final_loss(self, dir, final_test_loss, final_train_loss):
        final_test_loss = np.array([final_test_loss])
        final_train_loss = np.array([final_train_loss])
        filepath = f"{dir}/final_test_loss.txt"
        np.savetxt(filepath, final_test_loss, delimiter=",")
        filepath = f"{dir}/final_train_loss.txt"
        np.savetxt(filepath, final_train_loss, delimiter=",")


class ResidualBlock(nn.Module):
    dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool):
        h = nn.Dense(self.dim)(x)
        h = nn.silu(h)
        h = nn.Dense(self.dim)(h)
        h = nn.silu(h)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        return x + h


class PEZResidualMLP(nn.Module):
    feat_dim: int
    hidden_dim: int = 64
    n_blocks: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, features, deterministic=True):
        # features: (batch, feat_dim), logit_phi0: (batch, 1)
        h = nn.Dense(self.hidden_dim)(features)
        h = nn.silu(h)
        for _ in range(self.n_blocks):
            h = ResidualBlock(self.hidden_dim, self.dropout_rate)(h, deterministic)

        out = nn.Dense(1)(h)  # (batch, 1)
        return nn.sigmoid(out).squeeze(-1)  # (batch,)

    def save_model(self, dir):
        feat_dim = np.array([self.feat_dim])
        hidden_dim = np.array([self.hidden_dim])
        n_blocks = np.array([self.n_blocks])
        dropout_rate = np.array([self.dropout_rate])
        filepath = f"{dir}/pez_feat_dim.txt"
        np.savetxt(filepath, feat_dim, delimiter=",")
        filepath = f"{dir}/pez_hidden_dim.txt"
        np.savetxt(filepath, hidden_dim, delimiter=",")
        filepath = f"{dir}/pez_n_blocks.txt"
        np.savetxt(filepath, n_blocks, delimiter=",")
        filepath = f"{dir}/pez_dropout_rate.txt"
        np.savetxt(filepath, dropout_rate, delimiter=",")

    def load_model(self, dir):
        filepath = f"{dir}/pez_feat_dim.txt"
        feat_dim = np.loadtxt(filepath, delimiter=",")
        filepath = f"{dir}/pez_hidden_dim.txt"
        hidden_dim = np.loadtxt(filepath, delimiter=",")
        filepath = f"{dir}/pez_n_blocks.txt"
        n_blocks = np.loadtxt(filepath, delimiter=",")
        filepath = f"{dir}/pez_dropout_rate.txt"
        dropout_rate = np.loadtxt(filepath, delimiter=",")
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_blocks = int(n_blocks)
        self.dropout_rate = float(dropout_rate)
        return feat_dim, hidden_dim, n_blocks, dropout_rate

    def save_final_loss(self, dir, final_test_loss, final_train_loss):
        final_test_loss = np.array([final_test_loss])
        final_train_loss = np.array([final_train_loss])
        filepath = f"{dir}/final_test_loss.txt"
        np.savetxt(filepath, final_test_loss, delimiter=",")
        filepath = f"{dir}/final_train_loss.txt"
        np.savetxt(filepath, final_train_loss, delimiter=",")
