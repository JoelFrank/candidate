"""Load BiSBM-EASY generator by file path, not via ambiguous top-level `data`."""
from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_generator():
    path = ROOT / 'data/synthetic/generator_easy_project5.py'
    if not path.is_file():
        raise FileNotFoundError(f'BiSBM-EASY generator missing: {path}')
    spec = importlib.util.spec_from_file_location('project5_bisbm_easy_generator', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load BiSBM-EASY generator: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_easy(*args, **kwargs):
    return load_generator().generate(*args, **kwargs)
