# pytorch_impl/training/train_mlp.py

import argparse
import os
import sys
import time
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons, make_circles
from torch.utils.data import TensorDataset, DataLoader  # For batching
from tqdm import tqdm

# Assuming mlp.py is in ../models/ relative to this script's location
try:
    from ..models.mlp import MLP # For running with 'python -m ...'
except (ImportError, ValueError):
    # Fallback for running script directly
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.join(script_dir, '..')
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from pytorch_impl.models.mlp import MLP

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def visualize_predictions(
    model: MLP,
    data_split: Tuple[torch.Tensor, torch.Tensor], # Expect PyTorch Tensors
    plot_title: str = "Model Predictions and Decision Boundary",
    save_path: str = None
) -> None:
    """
    Visualizes the model's decision boundary and validation data predictions for 2D data.
    """
    model.eval() # Set model to evaluation mode
    X_val, y_val = data_split
    X_val_np = X_val.cpu().numpy() # Convert to NumPy for plotting
    y_val_np = y_val.cpu().numpy().squeeze()

    if X_val_np.shape[1] != 2:
        print("Visualization only supported for 2D input features.")
        return

    print("Generating visualization...")

    x_min, x_max = X_val_np[:, 0].min() - 0.5, X_val_np[:, 0].max() + 0.5
    y_min, y_max = X_val_np[:, 1].min() - 0.5, X_val_np[:, 1].max() + 0.5
    step = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    grid_points_np = np.c_[xx.ravel(), yy.ravel()]
    grid_points_torch = torch.tensor(grid_points_np, dtype=torch.float32).to(DEVICE)

    with torch.no_grad(): # Disable gradient calculations for inference
        Z_logits = model(grid_points_torch)
    Z_pred_np = (Z_logits.cpu().numpy() > 0).astype(np.int32) # Convert logits to classes
    Z_pred_np = Z_pred_np.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z_pred_np, cmap=plt.cm.RdBu, alpha=0.8)
    scatter = plt.scatter(X_val_np[:, 0], X_val_np[:, 1], c=y_val_np, cmap=plt.cm.RdBu,
                        edgecolor='k', s=30, marker='o')

    with torch.no_grad():
        model_preds_val_logits = model(X_val)
    model_preds_val = (model_preds_val_logits.cpu().numpy() > 0).astype(np.int32)
    misclassified_mask = (model_preds_val != y_val.cpu().numpy()).squeeze()

    plt.scatter(X_val_np[misclassified_mask, 0], X_val_np[misclassified_mask, 1],
                facecolors='none', edgecolors='lime', s=100, linewidths=1.5,
                label='Misclassified')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(plot_title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    handles, labels = scatter.legend_elements(prop="colors")
    if np.any(misclassified_mask):
        misclassified_handle = plt.Line2D([0], [0], marker='o', color='w', label='Misclassified',
                                           markerfacecolor='none', markeredgecolor='lime', markersize=10, linestyle='None')
        plt.legend(handles=handles + [misclassified_handle], labels=['Class 0', 'Class 1', 'Misclassified'])
    else:
        plt.legend(handles=handles, labels=['Class 0', 'Class 1'])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    plt.close()

def make_dataset(
    dataset_type: str = 'linear',
    n_samples: int = 1000,
    input_size: int = 10,
    noise: float = 0.1,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Create a synthetic dataset. Returns PyTorch Tensors. """
    np.random.seed(seed) # For scikit-learn and numpy generation
    if dataset_type == 'moons':
        X_np, y_np = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32).reshape(-1, 1)
    elif dataset_type == 'circles':
        X_np, y_np = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
        return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32).reshape(-1, 1)
    elif dataset_type == 'linear':
        X_np = np.random.randn(n_samples, input_size)
        true_weights = np.random.randn(input_size, 1)
        y_np = np.dot(X_np, true_weights) + noise * np.random.randn(n_samples, 1)
        return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

def split_dataset(
    X: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.8
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
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

# --- Loss Functions (using PyTorch's built-ins) ---
# MSELoss will be instantiated as nn.MSELoss()
# BCEWithLogitsLoss will be instantiated as nn.BCEWithLogitsLoss()

def evaluate(
    model: MLP,
    dataloader: DataLoader, # Use DataLoader for evaluation
    loss_module: nn.Module, # Pass instantiated loss module
    device: torch.device
) -> Dict[str, float]:
    """ Evaluate model on the given dataset split. """
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    is_classification = isinstance(loss_module, nn.BCEWithLogitsLoss)

    with torch.no_grad(): # Disable gradient calculations
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch) # Get logits
            loss = loss_module(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

            if is_classification:
                # For BCEWithLogitsLoss, outputs are logits. Convert to probs then classes.
                predicted_classes = (torch.sigmoid(outputs) > 0.5).float()
                correct_predictions += (predicted_classes == y_batch).sum().item()

    avg_loss = total_loss / total_samples
    metrics = {"loss": avg_loss}
    if is_classification:
        accuracy = correct_predictions / total_samples
        metrics["accuracy"] = accuracy
    return metrics

def train(
    model: MLP,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_module: nn.Module, # Pass instantiated loss module
    device: torch.device,
    epochs: int = 100,
    eval_every: int = 10
) -> Tuple[MLP, Dict[str, list]]:
    """ Train the MLP model. """
    model.to(device) # Move model to device

    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch": []
    }
    is_classification = isinstance(loss_module, nn.BCEWithLogitsLoss)
    print(f"Starting PyTorch training with loss: {loss_module.__class__.__name__}")

    for epoch in range(epochs):
        model.train() # Set model to training mode
        start_time = time.time()
        epoch_train_loss = 0.0
        num_train_samples = 0

        for X_batch, y_batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = loss_module(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * X_batch.size(0)
            num_train_samples += X_batch.size(0)

        avg_train_loss = epoch_train_loss / num_train_samples
        elapsed = time.time() - start_time

        metrics_history["train_loss"].append(avg_train_loss)
        metrics_history["epoch"].append(epoch + 1)

        log_str = f"Epoch {epoch+1}/{epochs} - {elapsed:.2f}s - train_loss: {avg_train_loss:.6f}"

        if (epoch + 1) % eval_every == 0 or epoch == 0 or epoch == epochs - 1:
            val_metrics = evaluate(model, val_dataloader, loss_module, device)
            val_loss = val_metrics["loss"]
            metrics_history["val_loss"].append(val_loss)
            log_str += f" - val_loss: {val_loss:.6f}"
            if "accuracy" in val_metrics:
                val_acc = val_metrics["accuracy"]
                metrics_history["val_accuracy"].append(val_acc)
                log_str += f" - val_accuracy: {val_acc:.4f}"
        print(log_str)

    return model, metrics_history

def save_model_pt(model: MLP, filepath: str) -> None:
    """ Save trained PyTorch model state_dict to a file. """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"PyTorch model saved to {filepath}")

def load_model_pt(filepath: str, model_class: type, **kwargs_model_init) -> MLP:
    """ Load PyTorch model state_dict from a file. """
    model = model_class(**kwargs_model_init)
    model.load_state_dict(torch.load(filepath, map_location=DEVICE)) # Ensure map_location
    model.to(DEVICE) # Move to device after loading
    model.eval() # Set to eval mode after loading
    print(f"PyTorch model loaded from {filepath}")
    return model

def main(args):
    torch.manual_seed(args.seed) # Set seed for PyTorch
    np.random.seed(args.seed)    # Set seed for NumPy (used in data generation)

    if args.dataset_type in ['moons', 'circles']:
        input_size = 2
        output_size = 1 # Logit for binary classification
        loss_module = nn.BCEWithLogitsLoss().to(DEVICE)
        task_type = 'classification'
        print(f"Configuring for {args.dataset_type} (Binary Classification - PyTorch)")
    elif args.dataset_type == 'linear':
        input_size = args.linear_input_size
        output_size = 1
        loss_module = nn.MSELoss().to(DEVICE)
        task_type = 'regression'
        print(f"Configuring for {args.dataset_type} (Regression - PyTorch)")
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    hidden_size = args.hidden_size
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    eval_every = args.eval_every
    noise = args.noise

    model_init_kwargs = {'input_size': input_size, 'hidden_size': hidden_size, 'output_size': output_size}
    model = MLP(**model_init_kwargs).to(DEVICE) # Move model to device
    print("\nPyTorch Model Structure:")
    print(model)

    X, y = make_dataset(
        dataset_type=args.dataset_type,
        n_samples=args.n_samples,
        input_size=input_size,
        noise=noise,
        seed=args.seed
    )
    dataset_splits = split_dataset(X, y, train_ratio=0.8)

    train_data = TensorDataset(dataset_splits["train"][0], dataset_splits["train"][1])
    val_data = TensorDataset(dataset_splits["val"][0], dataset_splits["val"][1])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    print(f"\nDataset '{args.dataset_type}' created (PyTorch):")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Input shape: {X.shape[1:]}")
    print(f"  Output shape: {y.shape[1:]}")
    print(f"  Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nStarting PyTorch training...")
    trained_model, metrics = train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_module,
        DEVICE,
        epochs=epochs,
        eval_every=eval_every
    )

    final_train_metrics = evaluate(trained_model, train_dataloader, loss_module, DEVICE)
    final_val_metrics = evaluate(trained_model, val_dataloader, loss_module, DEVICE)

    print("\nPyTorch Final evaluation:")
    print(f"  Training loss: {final_train_metrics['loss']:.6f}")
    if "accuracy" in final_train_metrics: print(f"  Training accuracy: {final_train_metrics['accuracy']:.4f}")
    print(f"  Validation loss: {final_val_metrics['loss']:.6f}")
    if "accuracy" in final_val_metrics: print(f"  Validation accuracy: {final_val_metrics['accuracy']:.4f}")

    save_dir = "./saved_models_pytorch" # Separate directory for PyTorch models
    save_path = os.path.join(save_dir, f"mlp_model_{args.dataset_type}.pt")
    save_model_pt(trained_model, save_path)

    if task_type == 'classification':
        plot_dir = "./plots_pytorch" # Separate directory for PyTorch plots
        plot_path = os.path.join(plot_dir, f"decision_boundary_{args.dataset_type}.png")
        visualize_predictions(
            model=trained_model,
            data_split=(dataset_splits["val"][0], dataset_splits["val"][1]), # Pass tensors directly
            plot_title=f"PyTorch MLP Decision Boundary ({args.dataset_type.capitalize()} Dataset)",
            save_path=plot_path
        )

    print("\nPyTorch Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PyTorch MLP model.")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main(args)