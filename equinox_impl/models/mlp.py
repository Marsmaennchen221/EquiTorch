import equinox as eqx
import jax


class MLP(eqx.Module):
    """Simple Multi-Layer Perceptron (MLP) with one hidden layer and ReLU activation function."""
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, output_size, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)

        self.fc1 = eqx.nn.Linear(input_size, hidden_size, key=key1)
        self.fc2 = eqx.nn.Linear(hidden_size, output_size, key=key2)

    def __call__(self, x):
        x = self.fc1(x)
        x = jax.nn.relu(x)
        x = self.fc2(x)
        return x # NOTE: Returning just logits for now, might need to add sigmoid for probabilities later