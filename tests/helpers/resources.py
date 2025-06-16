from pathlib import Path

# Test-resources root
_resources_root = Path(__file__).parent / ".." / "resources"

def resolve_resource_path(resource_path: str | Path) -> Path:
    """
    Resolve the given resource path.

    Args:
        resource_path: Relative resource path to wanted resource in test/resources folder

    Returns:
        Absolute resource path
    """
    return Path(_resources_root / resource_path).resolve()
