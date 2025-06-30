import ast
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
from syrupy import SnapshotAssertion

from equitorch import converters
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
        assert len(tree.body) == 5  # noqa: PLR2004
        assert compare_ast(tree.body[0], ast.Import(names=[ast.alias(name="jax", asname=None)]))
        assert compare_ast(tree.body[1], ast.Import(names=[ast.alias(name="jax.numpy", asname="jnp")]))
        assert compare_ast(tree.body[2], ast.Import(names=[ast.alias(name="jax.random", asname="jr")]))
        assert compare_ast(tree.body[3], ast.Import(names=[ast.alias(name="equinox", asname="eqx")]))

    @pytest.mark.parametrize(
        "pytorch_code",
        [
            "class NonPyTorchModule():pass",
            "from abc import ABCMeta\nclass NonPyTorchModule(ABCMeta):pass",
            "import torch.nn as nn\nclass TestClass(nn.Module):\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()\n\t\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\t\tself.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n\t\tself.fc = nn.Linear(32 * 7 * 7, num_classes)",
            "import torch.nn as nn\nfrom abc import ABCMeta\nclass TestClass(nn.Module, ABCMeta):\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()\n\t\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\t\tself.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n\t\tself.fc = nn.Linear(32 * 7 * 7, num_classes)",
            "import torch.nn as nn\nfrom abc import ABCMeta\nclass TestClass(ABCMeta, nn.Module):\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()\n\t\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\t\tself.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n\t\tself.fc = nn.Linear(32 * 7 * 7, num_classes)",
            "import torch.nn as nn\nclass TestClass(nn.Module):\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()",
            "import torch.nn as nn\nclass TestClass(nn.Module):\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()\n\t\tself.number = 40 + 2",
            "import torch.nn as nn\nclass TestClass(nn.Module):\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()\n\t\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\t\tself.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n\t\tself.fc = nn.Linear(32 * 7 * 7, num_classes)\n\t\tself.constant = 42",
            "import torch.nn as nn\nclass TestClass(nn.Module):\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()\n\t\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\t\ta = 2 + 40\n\t\tself.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n\t\tself.fc = nn.Linear(32 * 7 * 7, num_classes)",
            "import torch.nn as nn\nclass TestClass(nn.Module):\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()\n\t\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\t\tself.conv1 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)",
        ],
        ids=["NonPytorchModule", "ABCMeta", "3 torch Assignments", "3 torch Assignments + ABCMeta", "ABCMeta + 3 torch Assignments", "no Assignments", "no torch Assignments", "torch Assignments and non torch Assignments", "torch Assignments and non torch Assignments not to self", "multiple torch Assignments same name"],
    )
    def test_visit_ClassDef(self, pytorch_code: str) -> None:  # noqa: N802
        transformer = PyTorchToEquinoxTransformer()
        tree = ast.parse(pytorch_code)
        num_bases = -1
        is_pytorch_module = False
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                num_bases = len(node.bases)
                try:
                    is_pytorch_module = transformer._is_pytorch_module(node)
                except KeyError:
                    is_pytorch_module = False
                break
        new_tree = transformer.visit(tree)
        for node in new_tree.body:
            if isinstance(node, ast.ClassDef):
                assert len(node.bases) == num_bases
                if not is_pytorch_module:
                    return
                pytorch_base_index = transformer._get_pytorch_module_base_index(node)
                assert isinstance(node.bases[pytorch_base_index], ast.Attribute)
                assert node.bases[pytorch_base_index].attr == "Module"  # type: ignore[attr-defined]
                assert isinstance(node.bases[pytorch_base_index].value, ast.Name)  # type: ignore[attr-defined]
                assert node.bases[pytorch_base_index].value.id == "eqx"  # type: ignore[attr-defined]
                return
        pytest.fail("Body did not contain a ast.ClassDef")

    @pytest.mark.parametrize(
        ("pytorch_code", "init_method", "forward_method"),
        [
            ("import torch.nn as nn\nclass TestClass(nn.Module):pass", False, False),
            ("import torch.nn as nn\nclass TestClass(nn.Module):\n\tdef __init__():pass", True, False),
            ("import torch.nn as nn\nclass TestClass(nn.Module):\n\tdef forward():pass", False, True),
            ("import torch.nn as nn\nclass TestClass(nn.Module):\n\tdef __init__():pass\n\tdef forward():pass", True, True),
        ],
        ids=["no-method", "init", "forward", "init and forward"],
    )
    def test_visit_FunctionDef(self, pytorch_code: str, init_method: bool, forward_method: bool) -> None:  # noqa: N802 FBT001
        transformer = PyTorchToEquinoxTransformer()
        tree = ast.parse(pytorch_code)
        with patch.object(PyTorchToEquinoxTransformer, "_transform_init_method") as transform_init_method, patch.object(PyTorchToEquinoxTransformer, "_transform_forward_method") as transform_forward_method:
            transformer.visit(tree)
            transform_init_method.assert_called_once() if init_method else transform_init_method.assert_not_called()
            transform_forward_method.assert_called_once() if forward_method else transform_forward_method.assert_not_called()

    @pytest.mark.parametrize(
        ("pytorch_code", "call_index"),
        [
            ("", []),
            ("x = self.fc(x)", ["module"]),
            ("import torch.nn.functional as F\nF.sigmoid()", ["functional"]),
            ("import torch\nx = torch.Tensor()\nx.view()", ["method", "functional"]),
            ("import torch.nn as nn\nnn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)", ["module_init"]),
        ],
        ids=["no call", "module", "functional", "method and functional", "module_init"],
    )
    def test_visit_Call(self, pytorch_code: str, call_index: list[Literal["module", "functional", "method", "module_init"]]) -> None:  # noqa: N802
        transformer = PyTorchToEquinoxTransformer()
        tree = ast.parse(pytorch_code)
        with (patch.object(PyTorchToEquinoxTransformer, "_transform_module_call") as transform_module_call,
              patch.object(PyTorchToEquinoxTransformer, "_transform_functional_call") as transform_functional_call,
              patch.object(PyTorchToEquinoxTransformer, "_transform_method_call") as transform_method_call,
              patch.object(PyTorchToEquinoxTransformer, "_transform_module_init") as transform_module_init):
            transformer.visit(tree)
            transform_module_call.assert_called_once() if "module" in call_index else transform_module_call.assert_not_called()
            transform_functional_call.assert_called_once() if "functional" in call_index else transform_functional_call.assert_not_called()
            transform_method_call.assert_called_once() if "method" in call_index else transform_method_call.assert_not_called()
            transform_module_init.assert_called_once() if "module_init" in call_index else transform_module_init.assert_not_called()

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
        ("code", "pytorch_module_index"),
        [
            ("class EmptyClass:pass", -1),
            ("class EmptyClass():pass", -1),
            ("import torch\nclass PyTorchClass(torch.nn.Module):pass", 0),
            ("import torch as T\nclass PyTorchClass(T.nn.Module):pass", 0),
            ("import wrong as torch\nclass PyTorchClass(torch.nn.Module):pass", -1),
            ("from torch import nn\nclass PyTorchClass(nn.Module):pass", 0),
            ("import torch.nn\nclass PyTorchClass(nn.Module):pass", 0),
            ("from torch import nn as equinox\nclass PyTorchClass(equinox.Module):pass", 0),
            ("from torch.nn import Module\nclass PyTorchClass(Module):pass", 0),
            ("from torch.nn import Module as mod\nclass PyTorchClass(mod):pass", 0),
            ("import equinox\nclass EquinoxClass(equinox.Module):pass", -1),
            ("import equinox as nn\nclass EquinoxClass(nn.Module):pass", -1),
            ("from equinox import Module\nclass EquinoxClass(Module):pass", -1),
            ("from equinox import Module as mod\nclass EquinoxClass(mod):pass", -1),
            ("from abc import ABCMeta\nfrom torch.nn import Module\nclass PyTorchClass(ABCMeta, Module):pass", 1),
            ("from abc import ABCMeta\nfrom torch.nn import Module\nclass PyTorchClass(Module, ABCMeta):pass", 0),
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
    def test_get_pytorch_module_base_index(self, code: str, pytorch_module_index: int) -> None:
        tree = ast.parse(code)
        transformer = PyTorchToEquinoxTransformer()
        transformer.visit(ast.parse(code))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if pytorch_module_index == -1:
                    with pytest.raises(ValueError, match=r"Class does not inherit from torch\.nn\.Module"):
                        transformer._get_pytorch_module_base_index(node)
                else:
                    assert transformer._get_pytorch_module_base_index(node) == pytorch_module_index

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
            ("def __init__(self, in_channels, num_classes):\n\tsuper(SimpleCNN, self).__init__()\n\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\tself.conv1 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)", ["conv1"]),
        ],
        ids=["3 torch Assignments", "no Assignments", "no torch Assignments", "torch Assignments and non torch Assignments", "torch Assignments and non torch Assignments not to self", "multiple torch Assignments same name"],
    )
    def test_extract_parameters(self, code: str, names: list[str]) -> None:
        tree = ast.parse(code)
        func_def: ast.FunctionDef = tree.body[0]  # type: ignore[assignment]
        transformer = PyTorchToEquinoxTransformer()
        current_class_params = transformer._extract_parameters(func_def)
        assert len(current_class_params) == len(names)
        for name, param_body in current_class_params.items():
            assert name in names
            assert isinstance(param_body["type"], str)
            assert isinstance(param_body["call_node"], ast.Call)
            assert isinstance(param_body["call_node"].func, ast.Attribute)
            assert param_body["call_node"].func.attr == param_body["type"]

    @pytest.mark.parametrize(
        "code",
        [
            "class TestClass:\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()\n\t\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\t\tself.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n\t\tself.fc = nn.Linear(32 * 7 * 7, num_classes)",
            "class TestClass:\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()",
            "class TestClass:\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()\n\t\tself.number = 40 + 2",
            "class TestClass:\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()\n\t\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\t\tself.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n\t\tself.fc = nn.Linear(32 * 7 * 7, num_classes)\n\t\tself.constant = 42",
            "class TestClass:\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()\n\t\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\t\ta = 2 + 40\n\t\tself.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n\t\tself.fc = nn.Linear(32 * 7 * 7, num_classes)",
            "class TestClass:\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()\n\t\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\t\tself.conv1 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)",
        ],
        ids=["3 torch Assignments", "no Assignments", "no torch Assignments", "torch Assignments and non torch Assignments", "torch Assignments and non torch Assignments not to self", "multiple torch Assignments same name"],
    )
    def test_add_type_annotations(self, code: str) -> None:
        tree = ast.parse(code)
        class_def: ast.ClassDef = tree.body[0]  # type: ignore[assignment]
        func_def: ast.FunctionDef = class_def.body[0]  # type: ignore[assignment]
        transformer = PyTorchToEquinoxTransformer()
        current_class_params = transformer._extract_parameters(func_def)
        PyTorchToEquinoxTransformer._add_type_annotations(class_def, current_class_params)
        num_annotations = 0
        for annotation in class_def.body:
            if isinstance(annotation, ast.AnnAssign):
                assert annotation.target.id in current_class_params  # type: ignore[union-attr]
                annotation_info: dict[str, str | ast.Call] = current_class_params[annotation.target.id]  # type: ignore[union-attr]
                num_annotations += 1
                assert isinstance(annotation.annotation, ast.Attribute)
                assert annotation.annotation.attr == annotation_info["type"]
                assert isinstance(annotation.annotation.value, ast.Attribute)
                assert annotation.annotation.value.attr == "nn"
                assert isinstance(annotation.annotation.value.value, ast.Name)
                assert annotation.annotation.value.value.id == "eqx"
        assert num_annotations == len(current_class_params)

    @pytest.mark.parametrize(
        "code",
        [
            "import torch.nn as nn\nclass TestClass(nn.Module):\n\tdef __init__():pass",
            "import torch.nn as nn\nclass TestClass(nn.Module):\n\tdef __init__(self, in_channels, num_classes):\n\t\tsuper(SimpleCNN, self).__init__()\n\t\tself.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)\n\t\tself.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n\t\tself.fc = nn.Linear(32 * 7 * 7, num_classes)",
        ],
        ids=["empty init", "non-empty init"],
    )
    def test_transform_init_method(self, code: str) -> None:  # noqa: PLR0915
        tree = ast.parse(code)
        num_stmts = len(tree.body[-1].body[-1].body)  # type: ignore[attr-defined]
        transformer = PyTorchToEquinoxTransformer()
        transformer.visit(tree)
        num_keys = transformer.key_counter
        assert isinstance(tree.body[-1], ast.ClassDef)
        assert isinstance(tree.body[-1].body[-1], ast.FunctionDef)
        assert tree.body[-1].body[-1].name == "__init__"
        assert any(arg.arg == "key" for arg in tree.body[-1].body[-1].args.args)
        assert tree.body[-1].body[-1].args.kwarg
        assert tree.body[-1].body[-1].args.kwarg.arg == "kwargs"
        assert len(tree.body[-1].body[-1].body) == 2 + num_stmts

        # init key
        assert isinstance(tree.body[-1].body[-1].body[0], ast.If)
        assert isinstance(tree.body[-1].body[-1].body[0].test, ast.Compare)
        assert isinstance(tree.body[-1].body[-1].body[0].test.left, ast.Name)
        assert tree.body[-1].body[-1].body[0].test.left.id == "key"
        assert isinstance(tree.body[-1].body[-1].body[0].test.left.ctx, ast.Load)
        assert len(tree.body[-1].body[-1].body[0].test.ops) == 1
        assert isinstance(tree.body[-1].body[-1].body[0].test.ops[0], ast.Is)
        assert len(tree.body[-1].body[-1].body[0].test.comparators) == 1
        assert isinstance(tree.body[-1].body[-1].body[0].test.comparators[0], ast.Constant)
        assert tree.body[-1].body[-1].body[0].test.comparators[0].value is None
        assert len(tree.body[-1].body[-1].body[0].body) == 1
        assert isinstance(tree.body[-1].body[-1].body[0].body[0], ast.Assign)
        assert len(tree.body[-1].body[-1].body[0].body[0].targets) == 1
        assert isinstance(tree.body[-1].body[-1].body[0].body[0].targets[0], ast.Name)
        assert tree.body[-1].body[-1].body[0].body[0].targets[0].id == "key"
        assert isinstance(tree.body[-1].body[-1].body[0].body[0].targets[0].ctx, ast.Store)
        assert isinstance(tree.body[-1].body[-1].body[0].body[0].value, ast.Call)
        assert isinstance(tree.body[-1].body[-1].body[0].body[0].value.func, ast.Attribute)
        assert isinstance(tree.body[-1].body[-1].body[0].body[0].value.func.value, ast.Name)
        assert tree.body[-1].body[-1].body[0].body[0].value.func.value.id == "jr"
        assert isinstance(tree.body[-1].body[-1].body[0].body[0].value.func.value.ctx, ast.Load)
        assert tree.body[-1].body[-1].body[0].body[0].value.func.attr == "PRNGKey"
        assert isinstance(tree.body[-1].body[-1].body[0].body[0].value.func.ctx, ast.Load)
        assert len(tree.body[-1].body[-1].body[0].body[0].value.args) == 1
        assert isinstance(tree.body[-1].body[-1].body[0].body[0].value.args[0], ast.Constant)
        assert tree.body[-1].body[-1].body[0].body[0].value.args[0].value == 0

        # split key
        assert isinstance(tree.body[-1].body[-1].body[1], ast.Assign)
        assert len(tree.body[-1].body[-1].body[1].targets) == 1
        assert isinstance(tree.body[-1].body[-1].body[1].targets[0], ast.Tuple)
        assert len(tree.body[-1].body[-1].body[1].targets[0].elts) == num_keys
        for n in range(num_keys):
            assert isinstance(tree.body[-1].body[-1].body[1].targets[0].elts[n], ast.Name)
            assert tree.body[-1].body[-1].body[1].targets[0].elts[n].id == f"key_{n}"  # type: ignore[attr-defined]
            assert isinstance(tree.body[-1].body[-1].body[1].targets[0].elts[n].ctx, ast.Store)  # type: ignore[attr-defined]
        assert isinstance(tree.body[-1].body[-1].body[1].value, ast.Call)
        assert isinstance(tree.body[-1].body[-1].body[1].value.func, ast.Attribute)
        assert isinstance(tree.body[-1].body[-1].body[1].value.func.value, ast.Name)
        assert tree.body[-1].body[-1].body[1].value.func.value.id == "jr"
        assert isinstance(tree.body[-1].body[-1].body[1].value.func.value.ctx, ast.Load)
        assert tree.body[-1].body[-1].body[1].value.func.attr == "split"
        assert isinstance(tree.body[-1].body[-1].body[1].value.func.ctx, ast.Load)
        assert len(tree.body[-1].body[-1].body[1].value.args) == 2  # noqa: PLR2004
        assert isinstance(tree.body[-1].body[-1].body[1].value.args[0], ast.Name)
        assert tree.body[-1].body[-1].body[1].value.args[0].id == "key"
        assert isinstance(tree.body[-1].body[-1].body[1].value.args[0].ctx, ast.Load)
        assert isinstance(tree.body[-1].body[-1].body[1].value.args[1], ast.Constant)
        assert tree.body[-1].body[-1].body[1].value.args[1].value == num_keys

    def test_transform_forward_method(self) -> None:
        new_tree = transform_to_equinox_ast("def forward(self, x):\n\treturn x")
        assert isinstance(new_tree.body[-1], ast.FunctionDef)  # type: ignore[attr-defined]
        assert new_tree.body[-1].name == "__call__"  # type: ignore[attr-defined]

    def test_is_self_module_call(self) -> None:
        self_module_call = ast.parse("self.forward(x)")
        non_self_module_call = ast.parse("a.forward(x)")
        assert isinstance(self_module_call.body[0], ast.Expr)
        assert isinstance(non_self_module_call.body[0], ast.Expr)
        assert isinstance(self_module_call.body[0].value, ast.Call)
        assert isinstance(non_self_module_call.body[0].value, ast.Call)
        assert PyTorchToEquinoxTransformer._is_self_module_call(self_module_call.body[0].value)
        assert not PyTorchToEquinoxTransformer._is_self_module_call(non_self_module_call.body[0].value)

    @pytest.mark.parametrize(
        ("import_code", "code", "is_functional_call"),
        [
            ("import torch.nn.functional as F", "F.function(x)", True),
            ("import torch.nn.functional", "functional.function(x)", True),
            ("import wrong as W", "W.function(x)", False),
            ("import torch as T", "T.function(x)", True),
            ("import torch", "torch.function(x)", True),
            ("import wrong", "wrong.function(x)", False),
            ("import torch.nn as N", "N.functional.function(x)", True),
            ("import torch as T", "T.nn.functional.function(x)", True),
            ("import torch", "torch.nn.functional.function(x)", True),
            ("import torch", "torch.W.functional.function(x)", False),
            ("import torch", "torch.nn.W.function(x)", False),
            ("import torch.nn as N", "N.wrong.function(x)", False),
        ],
        ids=["torch.nn.functional as F", "torch.nn.functional", "wrong as W", "torch as T", "torch", "wrong", "torch.nn as N functional", "torch as T functional", "torch functional", "torch torch.W.functional", "torch torch.nn.W functional", "torch.nn as N wrong functional" ],
    )
    def test_is_functional_call(self, import_code: str, code: str, is_functional_call: bool) -> None:  # noqa: FBT001
        tree = ast.parse(code)
        import_tree = ast.parse(import_code)
        transformer = PyTorchToEquinoxTransformer()
        transformer.visit(import_tree)
        node_index = 0
        for i, node in enumerate(tree.body):  # Jump over import statements
            if isinstance(node, ast.Expr):
                node_index = i
        assert isinstance(tree.body[node_index].value, ast.Call)  # type: ignore[attr-defined]
        assert transformer._is_functional_call(tree.body[node_index].value) == is_functional_call  # type: ignore[attr-defined]

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
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                assert PyTorchToEquinoxTransformer._is_pytorch_module_init(node.value) == pytorch_module_init
            else:
               pytest.fail("Not all nodes are ast.Expr with node.value being ast.Call")

    @pytest.mark.parametrize(
        "code",
        [
            "x = self.function1(y)",
            "a = self.function2(b, c)",
        ],
        ids=["one arg", "two args"],
    )
    def test_transform_module_call(self, code: str) -> None:
        tree = ast.parse(code)
        target_name = tree.body[-1].targets[0].id  # type: ignore[attr-defined]
        module_function_name = tree.body[-1].value.func.attr  # type: ignore[attr-defined]
        module_function_variable_num = len(tree.body[-1].value.args)  # type: ignore[attr-defined]
        module_function_args = []
        for arg in tree.body[-1].value.args:  # type: ignore[attr-defined]
            module_function_args.append((type(arg), arg.id))
        transformer = PyTorchToEquinoxTransformer()
        transformer.visit(tree)
        assert isinstance(tree.body[-1], ast.Assign)
        assert len(tree.body[-1].targets) == 1
        assert isinstance(tree.body[-1].targets[0], ast.Name)
        assert tree.body[-1].targets[0].id == target_name
        assert isinstance(tree.body[-1].targets[0].ctx, ast.Store)
        assert isinstance(tree.body[-1].value, ast.Call)
        assert isinstance(tree.body[-1].value.func, ast.Call)
        assert isinstance(tree.body[-1].value.func.func, ast.Attribute)
        assert tree.body[-1].value.func.func.attr == "vmap"
        assert isinstance(tree.body[-1].value.func.func.value, ast.Name)
        assert tree.body[-1].value.func.func.value.id == "jax"
        assert len(tree.body[-1].value.func.args) == 1
        assert isinstance(tree.body[-1].value.func.args[0], ast.Attribute)
        assert tree.body[-1].value.func.args[0].attr == module_function_name
        assert isinstance(tree.body[-1].value.func.args[0].value, ast.Name)
        assert tree.body[-1].value.func.args[0].value.id == "self"
        assert len(tree.body[-1].value.args) == module_function_variable_num
        for i, arg in enumerate(tree.body[-1].value.args):
            assert isinstance(arg, module_function_args[i][0])
            assert arg.id == module_function_args[i][1]

    @pytest.mark.parametrize(
        ("code", "module_name", "function_name"),
        [
            ("import torch.nn.functional as F\nF.sigmoid()", "F", "sigmoid"),
            ("import torch\ntorch.Tensor()", "torch", "Tensor"),
        ],
        ids=["nn.functional", "torch"],
    )
    def test_transform_functional_call(self, code: str, module_name: str, function_name: str) -> None:
        tree = ast.parse(code)
        transformer = PyTorchToEquinoxTransformer()
        call = tree.body[-1].value  # type: ignore[attr-defined]
        while isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Call):
            call = call.func.value

        with patch.object(converters, "transform_functional_call") as transform_functional_call:
            transformer.visit(tree)
            transform_functional_call.assert_called_once_with(call, module_name, function_name)

    @pytest.mark.parametrize(
        ("code", "method_name"),
        [
            ("import torch\nx = torch.Tensor()\nx.size()", "size"),
            ("import torch\nx = torch.Tensor()\nx.reshape()", "reshape"),
            ("import torch\nx = torch.Tensor()\nx.flatten()", "flatten"),
        ],
        ids=["size", "reshape", "flatten"],
    )
    def test_transform_method_call(self, code: str, method_name: str) -> None:
        tree = ast.parse(code)
        transformer = PyTorchToEquinoxTransformer()
        call = tree.body[-1].value  # type: ignore[attr-defined]

        with patch.object(converters, "transform_method_call") as transform_method_call:
            transformer.visit(tree)
            transform_method_call.assert_called_once_with(call, method_name)

    @pytest.mark.parametrize(
        ("code", "num_of_modules"),
        [
            ("import torch.nn as nn\nnn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)", 1),
            ("import torch.nn as nn\nnn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\nnn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)", 2),
        ],
        ids=["one module", "two modules"],
    )
    def test_transform_module_init(self, code: str, num_of_modules: int) -> None:
        tree = ast.parse(code)
        transformer = PyTorchToEquinoxTransformer()
        transformer.visit(tree)
        assert not isinstance(tree.body[-num_of_modules-1], ast.Expr)  # No more modules than expected
        for module_index in range(num_of_modules):
            assert isinstance(tree.body[-num_of_modules+module_index], ast.Expr)
            assert isinstance(tree.body[-num_of_modules+module_index].value, ast.Call)  # type: ignore[attr-defined]
            assert tree.body[-num_of_modules+module_index].value.keywords[-1].arg == "key"  # type: ignore[attr-defined]
            assert isinstance(tree.body[-num_of_modules+module_index].value.keywords[-1].value, ast.Name)  # type: ignore[attr-defined]
            assert tree.body[-num_of_modules+module_index].value.keywords[-1].value.id == f"key_{module_index}"  # type: ignore[attr-defined]

    def test_transform_module_init_reset_key_counter(self) -> None:
        code = ("import torch.nn as nn\n"
                "nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n"
                "nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n"
                "class NewClass(nn.Module):\n"
                "\tnn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)\n"
                "\tnn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)")
        tree = ast.parse(code)
        transformer = PyTorchToEquinoxTransformer()
        transformer.visit(tree)
        assert not isinstance(tree.body[-4], ast.Expr)  # No more modules than expected
        for key_index, node_index in enumerate([-3, -2]):
            assert isinstance(tree.body[node_index], ast.Expr)
            assert isinstance(tree.body[node_index].value, ast.Call)  # type: ignore[attr-defined]
            assert tree.body[node_index].value.keywords[-1].arg == "key"  # type: ignore[attr-defined]
            assert isinstance(tree.body[node_index].value.keywords[-1].value, ast.Name)  # type: ignore[attr-defined]
            assert tree.body[node_index].value.keywords[-1].value.id == f"key_{key_index}"  # type: ignore[attr-defined]
        assert isinstance(tree.body[-1], ast.ClassDef)  # ClassDef should reset key counter
        for key_index, node_index in enumerate([-2, -1]):
            assert isinstance(tree.body[-1].body[node_index], ast.Expr)
            assert isinstance(tree.body[-1].body[node_index].value, ast.Call)  # type: ignore[attr-defined]
            assert tree.body[-1].body[node_index].value.keywords[-1].arg == "key"  # type: ignore[attr-defined]
            assert isinstance(tree.body[-1].body[node_index].value.keywords[-1].value, ast.Name)  # type: ignore[attr-defined]
            assert tree.body[-1].body[node_index].value.keywords[-1].value.id == f"key_{key_index}"  # type: ignore[attr-defined]

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
