import os
import sys
import time
from typing import Tuple, Dict, Any, Callable
import argparse

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax import random
import numpy as np
from tqdm import tqdm
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt

try:
    from ..models.mlp import MLP
except (ImportError, ValueError):
    # Fallback for running script directly
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from equinox_impl.models.mlp import MLP


def visualize_predictions(
    model: MLP,
    data_split: Tuple[jnp.ndarray, jnp.ndarray],
    plot_title: str = "Model Predictions and Decision Boundary",
    save_path: str = None
) -> None:
    """
    Visualizes the model's decision boundary and validation data predictions for 2D data.

    Args:
        model: The trained Equinox MLP model.
        data_split: Tuple containing validation (X_val, y_val).
        plot_title: Title for the plot.
        save_path: Path to save the figure. If None, displays the plot.
    """
    X_val, y_val = data_split
    if X_val.shape[1] != 2:
        print("Visualization only supported for 2D input features.")
        return

    print("Generating visualization...")

    # plot boundaries with padding
    x_min, x_max = X_val[:, 0].min() - 0.5, X_val[:, 0].max() + 0.5
    y_min, y_max = X_val[:, 1].min() - 0.5, X_val[:, 1].max() + 0.5
    step = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    # Prepare grid points for prediction
    grid_points = jnp.c_[xx.ravel(), yy.ravel()]

    Z_logits = jax.vmap(model)(grid_points)

    # Convert logits to predicted classes (0 or 1)
    Z_pred = (Z_logits > 0).astype(jnp.int32)
    Z_pred = Z_pred.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    # Plot decision boundary with regions
    plt.contourf(xx, yy, Z_pred, cmap=plt.cm.RdBu, alpha=0.8)

    # Plot validation data points
    y_val_plot = np.array(y_val).squeeze()
    scatter = plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val_plot, cmap=plt.cm.RdBu,
                        edgecolor='k', s=30, marker='o')

    # Optionally Highlight misclassified points
    model_preds_val = (jax.vmap(model)(X_val) > 0).astype(jnp.int32)
    misclassified_mask = (model_preds_val != y_val).squeeze()
    plt.scatter(X_val[misclassified_mask, 0], X_val[misclassified_mask, 1],
                facecolors='none', edgecolors='lime', s=100, linewidths=1.5,
                label='Misclassified')


    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(plot_title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Add legend
    handles, labels = scatter.legend_elements(prop="colors")
    # Add misclassified marker if any exist
    if np.any(misclassified_mask):
        # Find the automatically generated handle for misclassified if possible, otherwise create one
         misclassified_handle = plt.Line2D([0], [0], marker='o', color='w', label='Misclassified',
                                           markerfacecolor='none', markeredgecolor='lime', markersize=10, linestyle='None')

         plt.legend(handles=handles + [misclassified_handle], labels=['Class 0', 'Class 1', 'Misclassified'])
    else:
        plt.legend(handles=handles, labels=['Class 0', 'Class 1'])


    # Save or show plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def make_dataset(
    key: jax.random.PRNGKey,
    dataset_type: str = 'linear',
    n_samples: int = 1000,
    input_size: int = 10,
    noise: float = 0.1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create a synthetic dataset.

    Args:
        key: JAX PRNG key
        dataset_type: Type of dataset ('linear', 'moons', 'circles')
        n_samples: Number of samples to generate
        input_size: Dimension of input features (only for 'linear')
        noise: Noise level for the dataset

    Returns:
        Tuple of (inputs, targets) as JAX arrays
    """
    if dataset_type == 'moons':
        X_np, y_np = make_moons(n_samples=n_samples, noise=noise, random_state=int(key[0]))
        return jnp.array(X_np), jnp.array(y_np).reshape(-1, 1)
    elif dataset_type == 'circles':
        X_np, y_np = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=int(key[0]))
        return jnp.array(X_np), jnp.array(y_np).reshape(-1, 1)
    elif dataset_type == 'linear':
        key1, key2, key3 = random.split(key, 3)
        X = random.normal(key1, (n_samples, input_size))
        true_weights = random.normal(key2, (input_size, 1))
        y = jnp.dot(X, true_weights) + noise * random.normal(key3, (n_samples, 1))
        return X, y
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

def split_dataset(
    X: jnp.ndarray,
    y: jnp.ndarray,
    train_ratio: float = 0.8
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    """ Splits dataset into training and validation sets. """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_size = int(n_samples * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val)
    }

def mse_loss(
    model_output: jnp.ndarray,
    y: jnp.ndarray
) -> jnp.ndarray:
    """ Mean squared error loss function (for regression). """
    return jnp.mean((model_output - y) ** 2)

def binary_cross_entropy_loss(
    model_output_logits: jnp.ndarray,
    y: jnp.ndarray
) -> jnp.ndarray:
    """ Binary cross-entropy loss (for binary classification). """
    # Applies sigmoid internally for numerical stability
    return jnp.mean(optax.sigmoid_binary_cross_entropy(model_output_logits, y))

@eqx.filter_value_and_grad
def compute_loss_and_grad(
    model: MLP,
    X: jnp.ndarray,
    y: jnp.ndarray,
    loss_fn: Callable
) -> jnp.ndarray:
    """ Compute loss and gradients for model parameters. 
    
    Args:
        model: The Equinox model (e.g., MLP).
        X: Batch of input features.
        y: Batch of corresponding targets.
        loss_fn: Callable function to compute the loss (e.g., MSE, BCE).

    Returns:
        A tuple containing the scalar loss value and a PyTree of gradients
        for the model's trainable parameters. (Return type modified by decorator).
    """
    # For classification, model outputs logits
    # For regression, model outputs predictions
    # The loss_fn handles interpretation (e.g., applying sigmoid)
    pred_logits_or_values = jax.vmap(model)(X) 
    return loss_fn(pred_logits_or_values, y)

@eqx.filter_jit
def make_step(
    model: MLP,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    X: jnp.ndarray,
    y: jnp.ndarray,
    loss_fn: Callable 
) -> Tuple[MLP, Any, jnp.ndarray]:
    """ Execute one optimization step. """
    loss, grads = compute_loss_and_grad(model, X, y, loss_fn)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def evaluate(
    model: MLP,
    data_split: Tuple[jnp.ndarray, jnp.ndarray],
    loss_fn: Callable,
    batch_size: int = 128
) -> Dict[str, float]:
    """ Evaluate model on the given dataset split. """
    X, y = data_split
    n_samples = X.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    total_loss = 0.0

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        pred_logits_or_values = jax.vmap(model)(X_batch)
        loss = loss_fn(pred_logits_or_values, y_batch)
        total_loss += loss * (end_idx - start_idx)

    avg_loss = total_loss / n_samples
    metrics = {"loss": float(avg_loss)}

    # Accuracy calculation dependent on used loss function to potentially add different ones later. Might need refactoring at some point.
    if loss_fn == binary_cross_entropy_loss:
        all_preds_logits = jax.vmap(model)(X)
        # Convert logits to predictions (0 or 1)
        predicted_classes = (all_preds_logits > 0).astype(jnp.int32)
        accuracy = jnp.mean(predicted_classes == y.astype(jnp.int32))
        metrics["accuracy"] = float(accuracy)

    return metrics

def train(
    model: MLP,
    dataset: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    optimizer: optax.GradientTransformation,
    loss_fn: Callable,
    batch_size: int = 32,
    epochs: int = 100,
    eval_every: int = 10,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)
) -> Tuple[MLP, Dict[str, list]]:
    """ Train the MLP model. """
    X_train, y_train = dataset["train"]
    n_samples = X_train.shape[0]
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch": []
    }

    print(f"Starting training with loss: {loss_fn.__name__}")

    for epoch in range(epochs):
        start_time = time.time()
        key, subkey = random.split(key)

        # Shuffle training data using JAX key
        perm = random.permutation(subkey, n_samples)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        epoch_loss = 0.0
        for i in tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_indices = jnp.arange(i, min(i + batch_size, n_samples))
            X_batch = X_train_shuffled[batch_indices]
            y_batch = y_train_shuffled[batch_indices]

            model, opt_state, batch_loss = make_step(
                model, opt_state, optimizer, X_batch, y_batch, loss_fn
            )
            epoch_loss += batch_loss * len(batch_indices)

        avg_train_loss = epoch_loss / n_samples
        elapsed = time.time() - start_time

        # Evaluate
        if (epoch + 1) % eval_every == 0 or epoch == 0 or epoch == epochs - 1:
            val_metrics = evaluate(model, dataset["val"], loss_fn, batch_size=batch_size)
            val_loss = val_metrics["loss"]
            metrics["train_loss"].append(float(avg_train_loss))
            metrics["val_loss"].append(val_loss)
            metrics["epoch"].append(epoch + 1)

            log_str = f"Epoch {epoch+1}/{epochs} - {elapsed:.2f}s - train_loss: {avg_train_loss:.6f} - val_loss: {val_loss:.6f}"
            if "accuracy" in val_metrics:
                 val_acc = val_metrics["accuracy"]
                 metrics["val_accuracy"].append(val_acc)
                 log_str += f" - val_accuracy: {val_acc:.4f}"
            print(log_str)

        else:
             # Log training loss only if not evaluating
             print(f"Epoch {epoch+1}/{epochs} - {elapsed:.2f}s - train_loss: {avg_train_loss:.6f}")


    return model, metrics


def save_model(model: MLP, filepath: str) -> None:
    """ Save trained model to a file. """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    eqx.tree_serialise_leaves(filepath, model)
    print(f"Model saved to {filepath}")

def load_model(filepath: str, example_model: MLP) -> MLP:
     """ Load model from a file, requires example model for structure. """
     return eqx.tree_deserialise_leaves(filepath, example_model)


def main(args):
    """Main function to train the MLP model."""
    seed = args.seed
    key = random.PRNGKey(seed)

    # Determine parameters based on dataset
    if args.dataset_type in ['moons', 'circles']:
        input_size = 2 
        output_size = 1 
        loss_fn = binary_cross_entropy_loss
        task_type = 'classification'
        print(f"Configuring for {args.dataset_type} (Binary Classification)")
    elif args.dataset_type == 'linear':
        input_size = args.linear_input_size # Use specific arg for linear
        output_size = 1 # Single output for regression
        loss_fn = mse_loss
        task_type = 'regression'
        print(f"Configuring for {args.dataset_type} (Regression)")
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # Define general hyperparameters
    hidden_size = args.hidden_size
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    eval_every = args.eval_every
    noise = args.noise

    # Create model
    key, subkey = random.split(key)
    model = MLP(input_size, hidden_size, output_size, key=subkey)
    print("\nModel Structure:")
    print(model)

    # Create dataset
    key, data_key = random.split(key)
    X, y = make_dataset(
        data_key,
        dataset_type=args.dataset_type,
        n_samples=args.n_samples,
        input_size=input_size, # Pass correct input size
        noise=noise
    )
    dataset = split_dataset(X, y, train_ratio=0.8)
    print(f"\nDataset '{args.dataset_type}' created:")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Input shape: {X.shape[1:]}")
    print(f"  Output shape: {y.shape[1:]}")
    print(f"  Train size: {dataset['train'][0].shape[0]}, Val size: {dataset['val'][0].shape[0]}")


    # Define optimizer
    optimizer = optax.adam(learning_rate)

    # Train model
    print("\nStarting training...")
    key, train_key = random.split(key)
    trained_model, metrics = train(
        model,
        dataset,
        optimizer,
        loss_fn=loss_fn,
        batch_size=batch_size,
        epochs=epochs,
        eval_every=eval_every,
        key=train_key # Pass key for reproducible shuffling
    )

    # Evaluate final model
    final_train_metrics = evaluate(trained_model, dataset["train"], loss_fn, batch_size)
    final_val_metrics = evaluate(trained_model, dataset["val"], loss_fn, batch_size)

    print("\nFinal evaluation:")
    print(f"  Training loss: {final_train_metrics['loss']:.6f}")
    if "accuracy" in final_train_metrics: print(f"  Training accuracy: {final_train_metrics['accuracy']:.4f}")
    print(f"  Validation loss: {final_val_metrics['loss']:.6f}")
    if "accuracy" in final_val_metrics: print(f"  Validation accuracy: {final_val_metrics['accuracy']:.4f}")

    # Save model
    save_dir = "./saved_models"
    save_path = os.path.join(save_dir, f"mlp_model_{args.dataset_type}.eqx")
    save_model(trained_model, save_path)

    # Add Visualization Step
    if task_type == 'classification':
        plot_dir = "./plots"
        plot_path = os.path.join(plot_dir, f"decision_boundary_{args.dataset_type}.png")
        visualize_predictions(
            model=trained_model,
            data_split=dataset["val"], # Use validation split for visualization
            plot_title=f"MLP Decision Boundary ({args.dataset_type.capitalize()} Dataset)",
            save_path=plot_path
        )

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Equinox MLP model.")
    parser.add_argument("--dataset-type", type=str, default="moons",
                        choices=["linear", "moons", "circles"],
                        help="Type of dataset to generate.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples.")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level for dataset generation.")
    parser.add_argument("--linear-input-size", type=int, default=10, help="Input size for linear dataset.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate every N epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    main(args)