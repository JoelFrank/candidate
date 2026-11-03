"""
Synthetic BiSBM Inductive Data Loader

Loads the pre-generated .npz files produced by data/synthetic/generator.py
and returns a SyntheticInductiveData dataclass whose structure mirrors
EuroControlInductiveData closely enough that the training scripts only need
minimal adaptation.

Weight conditions (one npz per condition):
  informative      synthetic_informative.npz
  anti_correlated  synthetic_anti_correlated.npz
  orthogonal       synthetic_orthogonal.npz

Pair types stored per val/test edge: 'O-O', 'O-U', 'U-U'

Vocabulary convention (mirrors EuroControl):
  Index 0  = <UNK>  (new / unseen node in val/test)
  Index i  = old node i-1 (so old node 0 → vocab 1, old node k → vocab k+1)

Usage
-----
  from src.data.synthetic_loader import SyntheticInductiveLoader
  data = SyntheticInductiveLoader('data/synthetic', condition='informative').load()
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


# ── vocabulary ────────────────────────────────────────────────────────────────

@dataclass
class SyntheticVocabulary:
    """Vocabulary of known (train-visible) nodes.

    Index 0 is always reserved for <UNK> (new nodes in val/test).
    """
    num_u: int   # = N_U_OLD + 1 (includes UNK at 0)
    num_v: int   # = N_V_OLD + 1 (includes UNK at 0)
    unk_u_idx: int = 0
    unk_v_idx: int = 0


# ── main data container ───────────────────────────────────────────────────────

@dataclass
class SyntheticInductiveData:
    """
    Inductive-temporal data splits for a synthetic BiSBM graph.

    Field naming is kept parallel to EuroControlInductiveData so that
    training scripts only need minor renaming to work with both datasets.

      EuroControl field           Synthetic equivalent
      ─────────────────────────   ────────────────────────────
      train_airport_features      train_u_features
      train_airline_features      train_v_features
      train_airport_ids           train_u_ids   (vocab indices, 0=UNK)
      train_airline_ids           train_v_ids
      train_num_airports          train_num_u
      train_num_airlines          train_num_v
      vocabulary.num_airports     vocabulary.num_u
      vocabulary.num_airlines     vocabulary.num_v
      airport_feat_dim            feat_dim_u
      airline_feat_dim            feat_dim_v
      (no equivalent)             train_timestamps  (for temporal windows)
      (no equivalent)             val_pair_types    ('O-O'/'O-U'/'U-U')
      (no equivalent)             test_pair_types
    """

    # ── TRAIN (PAST – 80 % of O-O edges by timestamp) ──────────────────
    train_u_ids: np.ndarray             # vocab indices [N_train_u]
    train_v_ids: np.ndarray             # vocab indices [N_train_v]
    train_num_u: int
    train_num_v: int
    train_edge_index:   torch.LongTensor    # [2, E_train]  V offset by train_num_u
    train_edge_weights: torch.FloatTensor   # [E_train]
    train_u_features:   torch.FloatTensor   # [N_train_u, 4]
    train_v_features:   torch.FloatTensor   # [N_train_v, 3]
    train_timestamps:   torch.FloatTensor   # [E_train]  (for temporal windows)

    # ── VAL (PRESENT – 10 % O-O + inductive edges) ─────────────────────
    val_u_ids: np.ndarray
    val_v_ids: np.ndarray
    val_num_u: int
    val_num_v: int
    val_edge_index:   torch.LongTensor
    val_edge_weights: torch.FloatTensor
    val_u_features:   torch.FloatTensor
    val_v_features:   torch.FloatTensor
    val_pair_types:   np.ndarray            # ['O-O','O-U','U-U'] per edge

    # ── TEST (FUTURE – remainder O-O + inductive edges) ─────────────────
    test_u_ids: np.ndarray
    test_v_ids: np.ndarray
    test_num_u: int
    test_num_v: int
    test_edge_index:   torch.LongTensor
    test_edge_weights: torch.FloatTensor
    test_u_features:   torch.FloatTensor
    test_v_features:   torch.FloatTensor
    test_pair_types:   np.ndarray           # ['O-O','O-U','U-U'] per edge

    # ── vocabulary ──────────────────────────────────────────────────────
    vocabulary: SyntheticVocabulary

    # ── feature dimensions ──────────────────────────────────────────────
    feat_dim_u: int   # 4
    feat_dim_v: int   # 3

    # ── condition & block info ──────────────────────────────────────────
    weight_condition: str       # 'informative' | 'anti_correlated' | 'orthogonal'
    block_u: np.ndarray         # block assignment for each U node (global IDs)
    block_v: np.ndarray         # block assignment for each V node (global IDs)

    # ── statistics ──────────────────────────────────────────────────────
    stats: Dict[str, int]


# ── loader ────────────────────────────────────────────────────────────────────

class SyntheticInductiveLoader:
    """
    Loads a pre-generated synthetic_<condition>.npz file and returns a
    SyntheticInductiveData instance.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing the .npz files (default: data/synthetic/).
    condition : str
        One of 'informative', 'anti_correlated', 'orthogonal'.
    verbose : bool
    """

    VALID_CONDITIONS = {'informative', 'anti_correlated', 'orthogonal'}

    def __init__(
        self,
        data_dir: str = 'data/synthetic',
        condition: str = 'informative',
        verbose: bool = True,
    ):
        if condition not in self.VALID_CONDITIONS:
            raise ValueError(
                f"condition must be one of {self.VALID_CONDITIONS}, got '{condition}'"
            )
        self.data_dir  = Path(data_dir)
        self.condition = condition
        self.verbose   = verbose

    # ── public API ────────────────────────────────────────────────────────

    def load(self) -> SyntheticInductiveData:
        npz_path = self.data_dir / f'synthetic_{self.condition}.npz'

        if not npz_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {npz_path}\n"
                f"Run  python data/synthetic/generator.py  first."
            )

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  SYNTHETIC INDUCTIVE LOADER  condition={self.condition}")
            print(f"  File: {npz_path}")
            print(f"{'='*70}")

        d = np.load(npz_path, allow_pickle=True)

        # ── train ──────────────────────────────────────────────────────
        train_ei  = torch.from_numpy(d['train_edge_index']).long()
        train_ew  = torch.from_numpy(d['train_edge_weights']).float()
        train_uf  = torch.from_numpy(d['train_u_features']).float()
        train_vf  = torch.from_numpy(d['train_v_features']).float()
        train_ts  = torch.from_numpy(d['train_timestamps']).float()
        train_uid = d['train_u_ids']   # vocab indices (np.int64)
        train_vid = d['train_v_ids']
        n_train_u = int(train_uf.shape[0])
        n_train_v = int(train_vf.shape[0])

        # ── val ────────────────────────────────────────────────────────
        val_ei  = torch.from_numpy(d['val_edge_index']).long()
        val_ew  = torch.from_numpy(d['val_edge_weights']).float()
        val_uf  = torch.from_numpy(d['val_u_features']).float()
        val_vf  = torch.from_numpy(d['val_v_features']).float()
        val_uid = d['val_u_ids']
        val_vid = d['val_v_ids']
        val_pt  = d['val_pair_types']
        n_val_u = int(val_uf.shape[0])
        n_val_v = int(val_vf.shape[0])

        # ── test ───────────────────────────────────────────────────────
        test_ei  = torch.from_numpy(d['test_edge_index']).long()
        test_ew  = torch.from_numpy(d['test_edge_weights']).float()
        test_uf  = torch.from_numpy(d['test_u_features']).float()
        test_vf  = torch.from_numpy(d['test_v_features']).float()
        test_uid = d['test_u_ids']
        test_vid = d['test_v_ids']
        test_pt  = d['test_pair_types']
        n_test_u = int(test_uf.shape[0])
        n_test_v = int(test_vf.shape[0])

        # ── vocabulary ─────────────────────────────────────────────────
        n_u_old = int(d['n_u_old'][0])
        n_v_old = int(d['n_v_old'][0])
        vocab   = SyntheticVocabulary(
            num_u=n_u_old + 1,   # +1 for UNK at index 0
            num_v=n_v_old + 1,
        )

        # ── statistics ─────────────────────────────────────────────────
        stats = {
            'n_u_total'         : int(d['n_u_total'][0]),
            'n_v_total'         : int(d['n_v_total'][0]),
            'n_u_old'           : n_u_old,
            'n_v_old'           : n_v_old,
            'train_num_u'       : n_train_u,
            'train_num_v'       : n_train_v,
            'train_num_edges'   : int(train_ei.shape[1]),
            'val_num_u'         : n_val_u,
            'val_num_v'         : n_val_v,
            'val_num_edges'     : int(val_ei.shape[1]),
            'val_oo_edges'      : int((val_pt == 'O-O').sum()),
            'val_ou_edges'      : int((val_pt == 'O-U').sum()),
            'val_uu_edges'      : int((val_pt == 'U-U').sum()),
            'test_num_u'        : n_test_u,
            'test_num_v'        : n_test_v,
            'test_num_edges'    : int(test_ei.shape[1]),
            'test_oo_edges'     : int((test_pt == 'O-O').sum()),
            'test_ou_edges'     : int((test_pt == 'O-U').sum()),
            'test_uu_edges'     : int((test_pt == 'U-U').sum()),
        }

        if self.verbose:
            self._print_stats(stats, train_ew)

        return SyntheticInductiveData(
            # train
            train_u_ids        = train_uid,
            train_v_ids        = train_vid,
            train_num_u        = n_train_u,
            train_num_v        = n_train_v,
            train_edge_index   = train_ei,
            train_edge_weights = train_ew,
            train_u_features   = train_uf,
            train_v_features   = train_vf,
            train_timestamps   = train_ts,
            # val
            val_u_ids          = val_uid,
            val_v_ids          = val_vid,
            val_num_u          = n_val_u,
            val_num_v          = n_val_v,
            val_edge_index     = val_ei,
            val_edge_weights   = val_ew,
            val_u_features     = val_uf,
            val_v_features     = val_vf,
            val_pair_types     = val_pt,
            # test
            test_u_ids         = test_uid,
            test_v_ids         = test_vid,
            test_num_u         = n_test_u,
            test_num_v         = n_test_v,
            test_edge_index    = test_ei,
            test_edge_weights  = test_ew,
            test_u_features    = test_uf,
            test_v_features    = test_vf,
            test_pair_types    = test_pt,
            # vocabulary
            vocabulary         = vocab,
            # dims
            feat_dim_u         = int(train_uf.shape[1]),
            feat_dim_v         = int(train_vf.shape[1]),
            # condition
            weight_condition   = self.condition,
            block_u            = d['block_u'],
            block_v            = d['block_v'],
            # stats
            stats              = stats,
        )

    # ── private helpers ────────────────────────────────────────────────────

    def _print_stats(self, stats: dict, train_ew: torch.Tensor):
        print(f"\n[STATS]")
        for k, v in stats.items():
            print(f"  {k:<25}: {v}")
        print(f"\n[WEIGHTS] train edge weights:")
        print(f"  range  : [{train_ew.min():.2f}, {train_ew.max():.2f}]")
        print(f"  mean   : {train_ew.mean():.2f}  std: {train_ew.std():.2f}")
