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

    @pytest.mark.parametrize(
        ("code", "expected_import_name", "expected_import_value"),
        [
            ("import torch", "torch", "torch"),
            ("import torch as T", "T", "torch"),
            ("import torch.nn", "nn", "torch.nn"),
            ("import torch.nn as T", "T", "torch.nn"),
        ]
    )
    def test_visit_Import(self, code: str, expected_import_name: str, expected_import_value: str) -> None:  # noqa: N802
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

    @pytest.mark.parametrize(
        ("code", "pytorch_module"),
        [
            ("class EmptyClass:pass", False),
            ("class EmptyClass():pass", False),
            ("import torch\nclass PyTorchClass(torch.nn.Module):pass", True),
            ("import torch as T\nclass PyTorchClass(T.nn.Module):pass", True),
            ("import wrong as torch\nclass PyTorchClass(torch.nn.Module):pass", False),
            ("from torch import nn\nclass PyTorchClass(nn.Module):pass", True),
            ("import torch.nn\nclass PyTorchClass(nn.Module):pass", True),
            ("from torch import nn as equinox\nclass PyTorchClass(equinox.Module):pass", True),
            ("from torch.nn import Module\nclass PyTorchClass(Module):pass", True),
            ("from torch.nn import Module as mod\nclass PyTorchClass(mod):pass", True),
            ("import equinox\nclass EquinoxClass(equinox.Module):pass", False),
            ("import equinox as nn\nclass EquinoxClass(nn.Module):pass", False),
            ("from equinox import Module\nclass EquinoxClass(Module):pass", False),
            ("from equinox import Module as mod\nclass EquinoxClass(mod):pass", False),
            ("from abc import ABCMeta\nfrom torch.nn import Module\nclass PyTorchClass(ABCMeta, Module):pass", True),
            ("from abc import ABCMeta\nfrom torch.nn import Module\nclass PyTorchClass(Module, ABCMeta):pass", True),
        ],
        ids=[
            "EmptyClass",
            "EmptyClass()",
            "torch import",
            "torch import Alias",
            "torch import wrong Alias",
            "nn import",
            "torch.nn import",
            "nn import Alias",
            "torch Module import",
            "torch Module import Alias",
            "equinox import",
            "equinox as nn",
            "equinox Module import",
            "equinox Alias",
            "ABCMeta plus torch Module",
            "torch Module plus ABCMeta",
        ]
    )
    def test_is_pytorch_module(self, code: str, pytorch_module: bool) -> None:
        tree = ast.parse(code)
        transformer = PyTorchToEquinoxTransformer()
        transformer.visit(ast.parse(code))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                assert transformer._is_pytorch_module(node) == pytorch_module

    def test_find_init_method(self) -> None:
        assert PyTorchToEquinoxTransformer._find_init_method(ast.parse("")) is None
        assert PyTorchToEquinoxTransformer._find_init_method(ast.parse("class Test:pass").body[0]) is None
        assert PyTorchToEquinoxTransformer._find_init_method(ast.parse("class Test():pass").body[0]) is None
        assert PyTorchToEquinoxTransformer._find_init_method(ast.parse("class Test():\n\tdef test():pass").body[0]) is None
        init_method = PyTorchToEquinoxTransformer._find_init_method(ast.parse("class Test():\n\tdef __init__():pass").body[0])
        assert isinstance(init_method, ast.FunctionDef)
        assert init_method.name == "__init__"



def test_mlp_transpilation(snapshot_equinox_code: SnapshotAssertion):
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

    assert equinox_code == snapshot_equinox_code

def test_cnn_transpilation(snapshot_equinox_code: SnapshotAssertion):
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

    assert equinox_cnn_code == snapshot_equinox_code
