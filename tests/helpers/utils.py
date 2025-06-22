import ast
import itertools
from pathlib import Path
from typing import Any

from tests.helpers.resources import resolve_resource_path


def read_python_code(resource_path: str | Path) -> str:
    """Read the python code from a given resource_path.

    Args:
        resource_path: Relative resource path to python code in test/resources folder

    Returns:
        Python code

    Raises:
        ValueError:
            If the given resource is not a valid python file
    """
    absolute_path = resolve_resource_path(resource_path)
    if not absolute_path.is_file() or not absolute_path.suffix == ".py":
        raise ValueError("Resource is not a valid Python file")
    return absolute_path.read_text()


def compare_ast(tree1: ast.AST, tree2: ast.AST) -> bool:
    """Compare two ast trees.

    Args:
        tree1: The first ast tree
        tree2: The second ast tree

    Returns:
        bool: True if trees are equal, False otherwise
    """
    # Extra method to have typehints in public method
    return _compare_ast_recursive(tree1, tree2)


def _compare_ast_recursive(tree1: Any, tree2: Any) -> bool:
    """Compare two ast trees.

    Args:
        tree1: The first ast tree
        tree2: The second ast tree

    Returns:
        bool: True if trees are equal, False otherwise
    """
    if type(tree1) is not type(tree2):
        return False
    if isinstance(tree1, ast.AST):
        for k, v in vars(tree1).items():
            if k in ("lineno", "end_lineno", "col_offset", "end_col_offset", "ctx"):
                continue
            if not _compare_ast_recursive(v, getattr(tree2, k)):
                return False
        return True
    if isinstance(tree1, list):
        return all(itertools.starmap(_compare_ast_recursive, zip(tree1, tree2, strict=False)))
    return tree1 == tree2