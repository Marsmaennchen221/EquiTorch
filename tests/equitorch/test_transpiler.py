import ast
from pathlib import Path
from typing import List

import pytest
from syrupy import SnapshotAssertion

from equitorch.transpiler import PyTorchToEquinoxTransformer, transpile
from tests.helpers.utils import compare_ast, read_python_code


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
    ids=["mlp", "cnn"],
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
        ],
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
        ],
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
        assert len(tree.body) == 4  # noqa: PLR2004
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
        ],
    )
    def test_is_pytorch_module(self, code: str, pytorch_module: bool) -> None:  # noqa: FBT001
        tree = ast.parse(code)
        transformer = PyTorchToEquinoxTransformer()
        transformer.visit(ast.parse(code))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                assert transformer._is_pytorch_module(node) == pytorch_module

    @pytest.mark.parametrize(
        ("code", "pytorch_module_init"),
        [
            ("nn.Linear(32 * 7 * 7, 3)", True),
            ("nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)", True),
            ("nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)", True),
            ("nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)", True),
            ("nn.BatchNorm1d(num_features=100)", True),
            ("nn.BatchNorm2d(num_features=100)", True),
            ("nn.BatchNorm3d(num_features=100)", True),
            ("nn.Dropout(p=0.2)", True),
            ("nn.Sequential(nn.Conv2d(1,20,5))", False),
            ("ast.Load()", False),
        ],
        ids=["Linear", "Conv1d", "Conv2d", "Conv3d", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "Dropout", "not supported Sequential", "not a pytorch module"],
    )
    def test_is_pytorch_module_init(self, code: str, pytorch_module_init: bool) -> None:  # noqa: FBT001
        tree = ast.parse(code)
        print(ast.dump(tree, indent=4))
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                assert PyTorchToEquinoxTransformer._is_pytorch_module_init(node.value) == pytorch_module_init
            else:
               assert False

    def test_find_init_method(self) -> None:
        assert PyTorchToEquinoxTransformer._find_init_method(ast.parse("")) is None  # type: ignore[arg-type]
        assert PyTorchToEquinoxTransformer._find_init_method(ast.parse("class Test:pass").body[0]) is None  # type: ignore[arg-type]
        assert PyTorchToEquinoxTransformer._find_init_method(ast.parse("class Test():pass").body[0]) is None  # type: ignore[arg-type]
        assert PyTorchToEquinoxTransformer._find_init_method(ast.parse("class Test():\n\tdef test():pass").body[0]) is None  # type: ignore[arg-type]
        init_method = PyTorchToEquinoxTransformer._find_init_method(ast.parse("class Test():\n\tdef __init__():pass").body[0])  # type: ignore[arg-type]
        assert isinstance(init_method, ast.FunctionDef)
        assert init_method.name == "__init__"

    @pytest.mark.parametrize(
        ("code", "names"),
        [
            ("def __init__(self, in_channels, num_classes):\n\tsuper(SimpleCNN, self).__init__()\n\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\tself.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n\tself.fc = nn.Linear(32 * 7 * 7, num_classes)", ["conv1", "conv2", "fc"]),
            ("def __init__(self, in_channels, num_classes):\n\tsuper(SimpleCNN, self).__init__()", []),
            ("def __init__(self, in_channels, num_classes):\n\tsuper(SimpleCNN, self).__init__()\n\tself.number = 40 + 2", []),
            ("def __init__(self, in_channels, num_classes):\n\tsuper(SimpleCNN, self).__init__()\n\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\tself.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n\tself.fc = nn.Linear(32 * 7 * 7, num_classes)\n\tself.constant = 42", ["conv1", "conv2", "fc"]),
            ("def __init__(self, in_channels, num_classes):\n\tsuper(SimpleCNN, self).__init__()\n\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\ta = 2 + 40\n\tself.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n\tself.fc = nn.Linear(32 * 7 * 7, num_classes)", ["conv1", "conv2", "fc"]),
            ("def __init__(self, in_channels, num_classes):\n\tsuper(SimpleCNN, self).__init__()\n\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\tself.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)", ["conv1", "conv1"]),  # TODO: is this behaviour expected???
        ],
        ids=["3 torch Assignments", "no Assignments", "no torch Assignments", "torch Assignments and non torch Assignments", "torch Assignments and non torch Assignments not to self", "multiple torch Assignments same name"],
    )
    def test_extract_parameters(self, code: str, names: List[str]) -> None:
        tree = ast.parse(code)
        func_def: ast.FunctionDef = tree.body[0]  # type: ignore[assignment]
        transformer = PyTorchToEquinoxTransformer()
        current_class_params = transformer._extract_parameters(func_def)
        assert len(current_class_params) == len(names)
        for i, name in enumerate(names):
            assert isinstance(current_class_params[i]["name"], str)
            assert current_class_params[i]["name"] == name
            assert isinstance(current_class_params[i]["type"], str)
            assert isinstance(current_class_params[i]["call_node"], ast.Call)
            assert isinstance(current_class_params[i]["call_node"].func, ast.Attribute)
            assert current_class_params[i]["call_node"].func.attr == current_class_params[i]["type"]


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
