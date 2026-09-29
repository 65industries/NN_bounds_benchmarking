"""Hash-verified vendored sources; no Git or user-cache runtime dependency."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def verify_vendor(component, root=None, expected_commit=None):
    path = ROOT / 'vendor' / component
    if root is not None and Path(root).resolve() != path.resolve():
        raise ValueError('This reproduction uses only its bundled, pinned sources')
    manifest = json.loads((ROOT / 'vendor/sources.json').read_text())[component]
    if expected_commit is not None and manifest['commit'] != expected_commit:
        raise ValueError(f'Wrong source pin for {component}')
    for relative, expected in manifest['files'].items():
        source = path / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Vendored source changed: {component}/{relative}')
    actual = {p.relative_to(path).as_posix() for p in path.rglob('*.py')}
    expected_py = {p for p in manifest['files'] if p.endswith('.py')}
    if actual != expected_py:
        raise ValueError(f'Unexpected/missing Python sources in {component}')
    return path, manifest
