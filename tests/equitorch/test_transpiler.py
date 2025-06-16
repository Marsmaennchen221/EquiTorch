import ast
from pathlib import Path

import pytest
from syrupy import SnapshotAssertion

from equitorch.transpiler import transpile, PyTorchToEquinoxTransformer
from tests.helpers.utils import read_python_code, compare_ast


def transform_to_equinox_ast(pytorch_code: str) -> ast.AST:
    tree = ast.parse(pytorch_code)
    transformer = PyTorchToEquinoxTransformer()
    return transformer.visit(tree)


@pytest.mark.parametrize(
    "resource_path",
    [
        "networks/pytorch_mlp_example.py",
        "networks/pytorch_cnn_example.py",
    ],
    ids=["mlp", "cnn"]
)
def test_nn_transpilation(resource_path: str | Path, snapshot_equinox_code: SnapshotAssertion) -> None:
    """Test transpilation of PyTorch neural networks to Equinox."""

    # Read PyTorch code
    pytorch_code = read_python_code(resource_path)

    # Transpile to Equinox code
    equinox_code = transpile(pytorch_code)

    # Compare with previous saved snapshot
    assert equinox_code == snapshot_equinox_code


class TestPyTorchToEquinoxTransformer:

    @pytest.mark.parametrize(
        ("code", "expected_import_name", "expected_import_value"),
        [
            ("from torch import nn", "nn", "torch.nn"),
            ("from torch import nn as T", "T", "torch.nn"),
            ("from torch.nn import Module", "Module", "torch.nn.Module"),
            ("from torch.nn import Module as M", "M", "torch.nn.Module"),
        ]
    )
    def test_visit_ImportFrom(self, code: str, expected_import_name: str, expected_import_value: str) -> None:  # noqa: N802
        transformer = PyTorchToEquinoxTransformer()
        transformer.visit(ast.parse(code))
        assert transformer.current_imports[expected_import_name] == expected_import_value
        assert len(transformer.current_imports.keys()) == 1


    def test_visit_Module(self) -> None:  # noqa: N802
        pytorch_code = "nn.Module"
        tree = transform_to_equinox_ast(pytorch_code)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 4
        assert compare_ast(tree.body[0], ast.Import(names=[ast.alias(name="jax", asname=None)]))
        assert compare_ast(tree.body[1], ast.Import(names=[ast.alias(name="jax.numpy", asname="jnp")]))
        assert compare_ast(tree.body[2], ast.Import(names=[ast.alias(name="equinox", asname="eqx")]))


    def test_find_init_method(self) -> None:
        assert PyTorchToEquinoxTransformer._find_init_method(ast.parse("")) is None
        assert PyTorchToEquinoxTransformer._find_init_method(ast.parse("class Test:pass").body[0]) is None
        assert PyTorchToEquinoxTransformer._find_init_method(ast.parse("class Test():pass").body[0]) is None
        assert PyTorchToEquinoxTransformer._find_init_method(ast.parse("class Test():\n\tdef test():pass").body[0]) is None
        init_method = PyTorchToEquinoxTransformer._find_init_method(ast.parse("class Test():\n\tdef __init__():pass").body[0])
        assert isinstance(init_method, ast.FunctionDef)
        assert init_method.name == "__init__"


# TODO: remove test once fine-graded tests are implemented
def test_mlp_transpilation() -> None:
    """Test transpilation of a PyTorch MLP to Equinox."""
    # Read the PyTorch MLP example code
    pytorch_code = read_python_code("networks/pytorch_mlp_example.py")

    # Transpile the PyTorch code to Equinox code
    equinox_code = transpile(pytorch_code)

    # Verify the transpiled code contains expected elements
    assert "import jax" in equinox_code
    assert "import jax.numpy as jnp" in equinox_code
    assert "import equinox as eqx" in equinox_code
    assert "class MLP(eqx.Module):" in equinox_code
    assert "fc1: eqx.nn.Linear" in equinox_code
    assert "fc2: eqx.nn.Linear" in equinox_code
    assert "def __call__(self, x):" in equinox_code
    assert "jax.nn.relu" in equinox_code
    assert "jax.vmap(self.fc1)" in equinox_code
    assert "jax.vmap(self.fc2)" in equinox_code


# TODO: remove test once fine-graded tests are implemented
def test_cnn_transpilation() -> None:
    """Test transpilation of a PyTorch CNN to Equinox."""
    # Read the PyTorch CNN example code
    pytorch_cnn_code = read_python_code("networks/pytorch_cnn_example.py")

    # Transpile the PyTorch code to Equinox code
    equinox_cnn_code = transpile(pytorch_cnn_code)

    # Verify the transpiled code contains expected elements
    assert "import jax" in equinox_cnn_code
    assert "import jax.numpy as jnp" in equinox_cnn_code
    assert "import equinox as eqx" in equinox_cnn_code
    assert "class SimpleCNN(eqx.Module):" in equinox_cnn_code
    assert "conv1: eqx.nn.Conv2d" in equinox_cnn_code
    assert "conv2: eqx.nn.Conv2d" in equinox_cnn_code
    assert "fc: eqx.nn.Linear" in equinox_cnn_code
    assert "def __call__(self, x):" in equinox_cnn_code
    assert "jax.nn.relu" in equinox_cnn_code
    assert "jax.nn.max_pool" in equinox_cnn_code
    assert "jnp.reshape" in equinox_cnn_code
