"""Thin uproot helpers shared by the readers."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import uproot


def tree_names(path) -> set:
    """Tree names in a ROOT file, with ROOT's ';N' write cycles stripped.

    WCSim and fiTQun files routinely carry several cycles of the same tree
    (TTree::AutoSave writes a new one); uproot resolves a bare name to the
    highest cycle, which is the complete tree, so the cycle is never part of the
    identity we care about here.
    """
    with uproot.open(path) as f:
        return {k.split(";")[0] for k in f.keys(recursive=False)}


def branches(path, tree) -> set:
    with uproot.open(path) as f:
        return set(f[tree].keys())


def read_tree(path, tree, columns=None) -> pd.DataFrame:
    with uproot.open(path) as f:
        obj = f[tree]
        if columns is None:
            return obj.arrays(library="pd")
        present = [c for c in columns if c in obj.keys()]
        return obj.arrays(present, library="pd")


def has_tree(path, tree) -> bool:
    try:
        return tree in tree_names(path)
    except Exception:
        return False


def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None
