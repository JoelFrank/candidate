# -*- coding: utf-8 -*-
"""
WBT-BGRL Training Script — SYNTHETIC BiSBM Dataset

Graph: U-nodes -- V-nodes bipartite graph (BiSBM)
       Edge weight condition: {informative | anti_correlated | orthogonal}

Architecture mirrors experiments/eurocontrol/train_wbtbgrl.py exactly.
The only structural changes vs. the EuroControl script:
  1. SyntheticInductiveLoader (+ SyntheticInductiveData) instead of EuroControl
  2. SyntheticFeatureEmbedder — same hybrid ID+feature design, simpler dims
  3. --condition argument selects the weight regime
  4. Evaluation reports global metrics on TEST (10% future) — same protocol as real datasets

Supports all 4 WBT-BGRL variants:
  WP_WB | WP_NWB | NWP_WB | NWP_NWB

Usage:
  python experiments/synthetic/train_wbtbgrl.py --condition informative --variant WP_WB
"""

import sys
import os
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Optional
from torch_geometric.utils import negative_sampling
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import argparse
import random

from src.data.synthetic_loader import SyntheticInductiveLoader, SyntheticInductiveData
from src.models.wbtbgrl import WBTBGRL
from src.evaluation.decoder import LinkPredictor


# ── reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[SEED] {seed}")


# ── feature embedder ───────────────────────────────────────────────────────────

class SyntheticFeatureEmbedder(nn.Module):
    """
    Hybrid ID-embedding + numerical-feature embedder for synthetic nodes.

    Mirrors EuroControlFeatureEmbedder but with generic U/V naming:
      [id_embed | num_features]  →  Linear + ReLU  →  output_dim

    Index 0 in both embedding tables is <UNK> (zero-initialized) so new
    nodes in val/test receive zero ID contribution; their numerical features
    are also zero (inductive protocol), so their input vector is all-zeros
    projected through the linear layer.
    """

    def __init__(
        self,
        num_u: int,            # vocabulary size U (incl. UNK at 0)
        num_v: int,            # vocabulary size V (incl. UNK at 0)
        feat_dim_u: int = 4,
        feat_dim_v: int = 3,
        id_embed_dim: int = 32,
        output_dim: int = 128,
    ):
        super().__init__()
        self.id_embed_dim = id_embed_dim

        self.embed_u = nn.Embedding(num_u, id_embed_dim)
        self.embed_v = nn.Embedding(num_v, id_embed_dim)

        self.proj_u = nn.Linear(id_embed_dim + feat_dim_u, output_dim)
        self.proj_v = nn.Linear(id_embed_dim + feat_dim_v, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embed_u.weight)
        nn.init.xavier_uniform_(self.embed_v.weight)
        with torch.no_grad():
            self.embed_u.weight[0].fill_(0.0)   # UNK → zero
            self.embed_v.weight[0].fill_(0.0)
        nn.init.xavier_uniform_(self.proj_u.weight)
        nn.init.zeros_(self.proj_u.bias)
        nn.init.xavier_uniform_(self.proj_v.weight)
        nn.init.zeros_(self.proj_v.bias)

    def forward(
        self,
        u_features: torch.FloatTensor,                  # [N_u, feat_dim_u]
        v_features: torch.FloatTensor,                  # [N_v, feat_dim_v]
        u_ids: Optional[torch.LongTensor] = None,       # [N_u] vocab indices
        v_ids: Optional[torch.LongTensor] = None,       # [N_v] vocab indices
    ):
        if u_ids is not None:
            id_u = self.embed_u(u_ids)
        else:
            id_u = torch.zeros(u_features.size(0), self.id_embed_dim,
                               device=u_features.device)

        if v_ids is not None:
            id_v = self.embed_v(v_ids)
        else:
            id_v = torch.zeros(v_features.size(0), self.id_embed_dim,
                               device=v_features.device)

        x_u = F.relu(self.proj_u(torch.cat([id_u, u_features], dim=1)))
        x_v = F.relu(self.proj_v(torch.cat([id_v, v_features], dim=1)))
        return x_u, x_v


# ── negative sampling helpers ──────────────────────────────────────────────────

def create_global_edge_set_for_split(
    data: SyntheticInductiveData,
    split: str,
) -> torch.Tensor:
    """
    Return the union of known positive edges (V-offset stripped) up to *split*.
    Mirror of the same function in train_wbtbgrl_eurocontrol.py.
    """
    def strip(ei, num_u):
        e = ei.clone(); e[1] = e[1] - num_u; return e

    if split == 'val':
        parts = [
            strip(data.train_edge_index, data.train_num_u),
            strip(data.val_edge_index,   data.val_num_u),
        ]
    elif split == 'test':
        parts = [
            strip(data.train_edge_index, data.train_num_u),
            strip(data.val_edge_index,   data.val_num_u),
            strip(data.test_edge_index,  data.test_num_u),
        ]
    else:
        raise ValueError(split)

    return torch.unique(torch.cat(parts, dim=1), dim=1)


def _sample_typed_negatives(
    known_no_offset: torch.Tensor,
    u_pool: np.ndarray,
    v_pool: np.ndarray,
    n: int,
    test_num_u: int,
    device,
    seed: int = 0,
) -> torch.Tensor:
    """
    Sample n negative edges from u_pool × v_pool, excluding known_no_offset.
    Returns [2, n] with V-offset restored (v += test_num_u).
    """
    if len(u_pool) == 0 or len(v_pool) == 0 or n == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    known_set = set(zip(known_no_offset[0].tolist(), known_no_offset[1].tolist()))
    rng = np.random.default_rng(seed)
    sampled, seen = [], set(known_set)

    for _ in range(n * 200):
        u = int(rng.choice(u_pool))
        v = int(rng.choice(v_pool))
        if (u, v) not in seen:
            sampled.append((u, v))
            seen.add((u, v))
        if len(sampled) == n:
            break

    if not sampled:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    arr = np.array(sampled, dtype=np.int64)
    ei = torch.tensor(arr.T, dtype=torch.long)
    ei[1] += test_num_u
    return ei.to(device)


# ── trainer ────────────────────────────────────────────────────────────────────

class WBTBGRLSyntheticTrainer:
    """
    WBT-BGRL trainer for the synthetic BiSBM dataset.

    Phases
    ------
    Phase 1 — SSL pretraining on TRAIN graph.
    Phase 2 — Frozen encoder + supervised decoder on VAL graph.
    Phase 3 — Evaluation on TEST graph (global + per pair-type).
    """

    def __init__(
        self,
        data: SyntheticInductiveData,
        model_variant: str = 'WP_WB',
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.data          = data
        self.model_variant = model_variant
        self.hidden_dim    = hidden_dim
        self.output_dim    = output_dim
        self.num_layers    = num_layers
        self.dropout       = dropout
        self.device        = device

        parts = model_variant.split('_')
        self.use_weights_pretraining     = (parts[0] == 'WP')
        self.use_weights_link_prediction = (parts[1] == 'WB')

        print(f"[MODEL] WBT-BGRL variant={model_variant}  condition={data.weight_condition}")
        print(f"        WP={self.use_weights_pretraining}  WB={self.use_weights_link_prediction}")

        # Edge weights from loader (TRAIN only — no leakage)
        if data.train_edge_weights is not None:
            self.train_edge_weight = data.train_edge_weights
        else:
            self.train_edge_weight = torch.ones(data.train_edge_index.size(1))

        # Feature embedder
        feature_dim = hidden_dim
        self.feature_embedder = SyntheticFeatureEmbedder(
            num_u        = data.vocabulary.num_u,
            num_v        = data.vocabulary.num_v,
            feat_dim_u   = data.feat_dim_u,
            feat_dim_v   = data.feat_dim_v,
            id_embed_dim = 32,
            output_dim   = feature_dim,
        ).to(device)

        # WBTBGRL model with WeightedGCNEncoder
        self.model = WBTBGRL(
            input_dim_u              = feature_dim,
            input_dim_v              = feature_dim,
            hidden_dim               = hidden_dim,
            output_dim               = output_dim,
            num_layers               = num_layers,
            dropout                  = dropout,
            use_weights_pretraining  = self.use_weights_pretraining,
            use_weights_link_prediction = self.use_weights_link_prediction,
        ).to(device)

        print(f"[DEVICE] {device}")
        print(f"[EMBEDDER] num_u={data.vocabulary.num_u}  num_v={data.vocabulary.num_v}")
        print(f"[ENCODER] hidden={hidden_dim}  output={output_dim}  layers={num_layers}")

    # ── helpers ────────────────────────────────────────────────────────────

    def _embed(self, u_feat, v_feat, u_ids=None, v_ids=None):
        return self.feature_embedder(u_feat, v_feat, u_ids, v_ids)

    def _get_embeddings(self, u_feat, v_feat, edge_index, edge_weight, num_u,
                        u_ids=None, v_ids=None):
        self.model.eval()
        self.feature_embedder.eval()
        with torch.no_grad():
            x_u, x_v = self._embed(u_feat, v_feat, u_ids, v_ids)
            h_u, h_v = self.model.online_encoder(x_u, x_v, edge_index, edge_weight, num_u)
        return h_u, h_v

    # ── Phase 1 ────────────────────────────────────────────────────────────

    def train_encoder(self, epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-5):
        print(f"\n{'='*70}")
        print("[PHASE 1] SSL PRETRAINING (TRAIN graph)")
        print(f"  edges={self.data.train_edge_index.size(1)}  "
              f"WP={self.use_weights_pretraining}")

        optimizer = AdamW(
            list(self.feature_embedder.parameters()) + list(self.model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )

        dev = self.device
        ei  = self.data.train_edge_index.to(dev)
        ew  = self.train_edge_weight.to(dev)
        uf  = self.data.train_u_features.to(dev)
        vf  = self.data.train_v_features.to(dev)
        uid = torch.LongTensor(self.data.train_u_ids).to(dev)
        vid = torch.LongTensor(self.data.train_v_ids).to(dev)
        ew_input = ew if self.use_weights_pretraining else None

        losses = []
        self.model.train(); self.feature_embedder.train()

        for epoch in tqdm(range(epochs), desc="Phase 1"):
            optimizer.zero_grad()
            x_u, x_v = self._embed(uf, vf, uid, vid)
            outputs  = self.model(x_u, x_v, ei, self.data.train_num_u, ew_input)
            loss     = self.model.compute_loss(outputs, num_nodes_u=self.data.train_num_u)
            loss.backward()
            optimizer.step()
            self.model._update_target_encoder()
            losses.append(loss.item())
            if (epoch + 1) % 50 == 0:
                print(f"  epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

        print(f"[OK] Phase 1 done  final_loss={losses[-1]:.4f}")
        return losses

    # ── Phase 2 ────────────────────────────────────────────────────────────

    def train_decoder(self, epochs: int = 100, lr: float = 1e-3,
                      weight_decay: float = 1e-5, patience: int = 20):
        print(f"\n{'='*70}")
        print("[PHASE 2] DECODER TRAINING (VAL graph)")
        print(f"  edges={self.data.val_edge_index.size(1)}  "
              f"WB={self.use_weights_link_prediction}")

        # Freeze encoder
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.feature_embedder.parameters():
            p.requires_grad = False
        self.model.eval(); self.feature_embedder.eval()

        decoder = LinkPredictor(
            input_dim  = self.output_dim * 2,
            hidden_dim = self.hidden_dim,
            output_dim = 1,
            num_layers = 2,
            dropout    = self.dropout,
        ).to(self.device)

        optimizer = AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = (nn.BCEWithLogitsLoss(reduction='none')
                     if self.use_weights_link_prediction
                     else nn.BCEWithLogitsLoss())

        dev = self.device
        val_ei  = self.data.val_edge_index.to(dev)
        val_uf  = self.data.val_u_features.to(dev)
        val_vf  = self.data.val_v_features.to(dev)
        val_uid = torch.LongTensor(self.data.val_u_ids).to(dev)
        val_vid = torch.LongTensor(self.data.val_v_ids).to(dev)

        # Log1p-compressed edge weights for WB loss (same as EuroControl)
        if self.use_weights_link_prediction:
            raw_w           = self.data.val_edge_weights.to(dev)
            val_edge_weights = torch.log1p(raw_w)
        else:
            val_edge_weights = None

        # Frozen embeddings for val graph
        # WP: use actual val edge weights (consistent with weighted pretraining)
        # NWP: use uniform weights (consistent with unweighted pretraining)
        if self.use_weights_pretraining and self.data.val_edge_weights is not None:
            enc_weights_val = self.data.val_edge_weights.to(dev)
        else:
            enc_weights_val = torch.ones(val_ei.size(1), device=dev)
        h_u_val, h_v_val = self._get_embeddings(
            val_uf, val_vf, val_ei,
            enc_weights_val,
            self.data.val_num_u,
            u_ids=val_uid, v_ids=val_vid,
        )
        all_emb = torch.cat([h_u_val, h_v_val], dim=0)
        emb_layer = nn.Embedding.from_pretrained(all_emb, freeze=True)

        known_val_edges = create_global_edge_set_for_split(self.data, 'val')
        best_val, patience_ctr, best_state = 0.0, 0, None
        losses, val_metrics = [], []
        decoder.train()

        for epoch in tqdm(range(epochs), desc="Phase 2"):
            optimizer.zero_grad()

            neg_ei = negative_sampling(
                edge_index      = known_val_edges,
                num_nodes       = (self.data.val_num_u, self.data.val_num_v),
                num_neg_samples = val_ei.size(1),
                method          = 'sparse',
            )
            neg_ei[1] = neg_ei[1] + self.data.val_num_u
            neg_ei    = neg_ei.to(dev)

            edges  = torch.cat([val_ei, neg_ei], dim=1)
            labels = torch.cat([
                torch.ones(val_ei.size(1), device=dev),
                torch.zeros(neg_ei.size(1), device=dev),
            ])
            emb   = torch.cat([emb_layer(edges[0]), emb_layer(edges[1])], dim=1)
            pred  = decoder(emb).squeeze()

            if self.use_weights_link_prediction and val_edge_weights is not None:
                w = torch.cat([val_edge_weights,
                               torch.ones(neg_ei.size(1), device=dev)])
                loss = (criterion(pred, labels) * w).mean()
            else:
                loss = criterion(pred, labels)

            loss.backward(); optimizer.step()
            losses.append(loss.item())

            # Val ROC-AUC every 5 epochs
            if (epoch + 1) % 5 == 0:
                decoder.eval()
                with torch.no_grad():
                    eval_neg = negative_sampling(
                        edge_index      = known_val_edges,
                        num_nodes       = (self.data.val_num_u, self.data.val_num_v),
                        num_neg_samples = val_ei.size(1),
                        method          = 'sparse',
                    )
                    eval_neg[1] = eval_neg[1] + self.data.val_num_u
                    eval_neg    = eval_neg.to(dev)

                    pp = torch.sigmoid(decoder(
                        torch.cat([emb_layer(val_ei[0]), emb_layer(val_ei[1])], dim=1)
                    ).squeeze()).cpu().numpy()
                    np_ = torch.sigmoid(decoder(
                        torch.cat([emb_layer(eval_neg[0]), emb_layer(eval_neg[1])], dim=1)
                    ).squeeze()).cpu().numpy()

                from sklearn.metrics import roc_auc_score as _roc
                val_score = _roc(
                    np.concatenate([np.ones(len(pp)), np.zeros(len(np_))]),
                    np.concatenate([pp, np_]),
                )

                val_metrics.append(val_score)
                if val_score > best_val:
                    best_val     = val_score
                    patience_ctr = 0
                    best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
                else:
                    patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"\n[EARLY STOP] epoch {epoch+1}  Val ROC-AUC={val_score:.4f}")
                    break
                decoder.train()

        if best_state:
            decoder.load_state_dict(best_state)
        print(f"[OK] Phase 2 done  best Val ROC-AUC={best_val:.4f}")
        return decoder, losses, val_metrics

    # ── Phase 3 ────────────────────────────────────────────────────────────

    def evaluate_test(self, decoder) -> dict:
        print(f"\n{'='*70}")
        print("[PHASE 3] TEST EVALUATION (FUTURE graph — 10%)")

        dev = self.device
        test_ei  = self.data.test_edge_index.to(dev)
        test_uf  = self.data.test_u_features.to(dev)
        test_vf  = self.data.test_v_features.to(dev)
        test_uid = torch.LongTensor(self.data.test_u_ids).to(dev)
        test_vid = torch.LongTensor(self.data.test_v_ids).to(dev)

        # WP: use actual test edge weights (consistent with weighted pretraining)
        # NWP: use uniform weights (consistent with unweighted pretraining)
        if self.use_weights_pretraining and self.data.test_edge_weights is not None:
            enc_weights_test = self.data.test_edge_weights.to(dev)
        else:
            enc_weights_test = torch.ones(test_ei.size(1), device=dev)
        h_u, h_v = self._get_embeddings(
            test_uf, test_vf, test_ei,
            enc_weights_test,
            self.data.test_num_u,
            u_ids=test_uid, v_ids=test_vid,
        )
        all_emb = torch.cat([h_u, h_v], dim=0)

        known_test_edges = create_global_edge_set_for_split(self.data, 'test')
        neg_ei = negative_sampling(
            edge_index      = known_test_edges,
            num_nodes       = (self.data.test_num_u, self.data.test_num_v),
            num_neg_samples = test_ei.size(1),
            method          = 'sparse',
        )
        neg_ei[1] = neg_ei[1] + self.data.test_num_u
        neg_ei    = neg_ei.to(dev)

        all_ei   = torch.cat([test_ei, neg_ei], dim=1)
        edge_emb = torch.cat([all_emb[all_ei[0]], all_emb[all_ei[1]]], dim=1)

        decoder.eval()
        with torch.no_grad():
            pred = torch.sigmoid(decoder(edge_emb).squeeze())

        pos_pred = pred[:test_ei.size(1)]
        neg_pred = pred[test_ei.size(1):]

        results = _compute_metrics(pos_pred, neg_pred, tag="TEST")

        # ── per-type breakdown with 1:1 typed negatives (O-O / O-U / U-U) ───
        pt = self.data.test_pair_types
        if pt is not None and len(pt) == len(pos_pred):
            # Identify old vs new local node indices (vocab 0 = UNK = new node)
            test_uid_arr = self.data.test_u_ids   # [test_num_u] vocab indices
            test_vid_arr = self.data.test_v_ids   # [test_num_v] vocab indices
            old_u = np.where(test_uid_arr > 0)[0]
            new_u = np.where(test_uid_arr == 0)[0]
            old_v = np.where(test_vid_arr > 0)[0]
            new_v = np.where(test_vid_arr == 0)[0]

            num_u = self.data.test_num_u

            for ptype, prefix in [('O-O', 'OO'), ('O-U', 'OU'), ('U-U', 'UU')]:
                mask = np.where(pt == ptype)[0]
                if len(mask) == 0:
                    continue
                n_type = len(mask)
                # 1:1 typed negatives — same count as positives per pair type
                # Hits@k is well-defined: neg_count (100-120) >= k=100 ✓
                n_neg_type = n_type

                if ptype == 'O-O':
                    typed_neg_ei = _sample_typed_negatives(
                        known_test_edges, old_u, old_v, n_neg_type, num_u, dev, seed=0)
                elif ptype == 'O-U':
                    # Count how many positives are old_u→new_v vs new_u→old_v
                    pos_u = test_ei[0, mask].cpu().numpy()
                    is_old_u = np.isin(pos_u, old_u)
                    n_old_u_new_v = int(is_old_u.sum())
                    n_new_u_old_v = n_type - n_old_u_new_v
                    # Scale proportionally so total negatives = n_neg_type
                    if n_type > 0:
                        n_neg_old_u = max(round(n_old_u_new_v * n_neg_type / n_type), 0) \
                                      if n_old_u_new_v > 0 else 0
                        n_neg_new_u = n_neg_type - n_neg_old_u
                    else:
                        n_neg_old_u, n_neg_new_u = 0, 0
                    parts_neg = []
                    if n_neg_old_u > 0:
                        parts_neg.append(_sample_typed_negatives(
                            known_test_edges, old_u, new_v, n_neg_old_u, num_u, dev, seed=1))
                    if n_neg_new_u > 0:
                        parts_neg.append(_sample_typed_negatives(
                            known_test_edges, new_u, old_v, n_neg_new_u, num_u, dev, seed=2))
                    valid = [p for p in parts_neg if p.size(1) > 0]
                    if not valid:
                        continue
                    typed_neg_ei = torch.cat(valid, dim=1)
                else:  # U-U
                    typed_neg_ei = _sample_typed_negatives(
                        known_test_edges, new_u, new_v, n_neg_type, num_u, dev, seed=3)

                if typed_neg_ei.size(1) == 0:
                    continue

                typed_neg_emb = torch.cat(
                    [all_emb[typed_neg_ei[0]], all_emb[typed_neg_ei[1]]], dim=1)
                decoder.eval()
                with torch.no_grad():
                    typed_neg_pred = torch.sigmoid(decoder(typed_neg_emb).squeeze())

                type_res = _compute_metrics(pos_pred[mask], typed_neg_pred,
                                            tag=f"TEST/{ptype}")
                for k, v in type_res.items():
                    results[f'{prefix}_{k}'] = v

        print(f"\n[RESULTS]")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

        return results


# ── metrics helper ─────────────────────────────────────────────────────────────

def _compute_metrics(pos_pred: torch.Tensor, neg_pred: torch.Tensor,
                     tag: str = '') -> dict:
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score, recall_score, precision_score)

    labels = torch.cat([torch.ones(len(pos_pred)), torch.zeros(len(neg_pred))]).cpu().numpy()
    scores = torch.cat([pos_pred, neg_pred]).cpu().numpy()

    roc    = roc_auc_score(labels, scores)
    ap     = average_precision_score(labels, scores)
    bp     = (scores >= 0.5).astype(int)
    f1     = f1_score(labels, bp, zero_division=0)
    rec    = recall_score(labels, bp, zero_division=0)
    prec   = precision_score(labels, bp, zero_division=0)

    metrics = {
        'ROC-AUC': roc, 'AP': ap,
        'F1-Score (t=0.5)': f1, 'Recall (t=0.5)': rec, 'Precision (t=0.5)': prec,
    }
    for k in [10, 20, 50, 100]:
        if len(neg_pred) >= k:
            kth  = torch.topk(neg_pred, k)[0][-1]
            hits = float((pos_pred > kth).sum()) / len(pos_pred)
        else:
            all_ = torch.cat([pos_pred, neg_pred])
            kth  = torch.topk(all_, min(k, len(all_)))[0][-1]
            hits = float((pos_pred >= kth).sum()) / len(pos_pred)
        metrics[f'Hits@{k}'] = hits

    return metrics


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='WBT-BGRL on Synthetic BiSBM')
    parser.add_argument('--condition', type=str, default='informative',
                        choices=['informative', 'anti_correlated', 'orthogonal'])
    parser.add_argument('--variant',        type=str, default='WP_WB',
                        choices=['WP_WB', 'WP_NWB', 'NWP_WB', 'NWP_NWB'])
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--epochs_encoder', type=int, default=200)
    parser.add_argument('--epochs_decoder', type=int, default=100)
    parser.add_argument('--hidden_dim',     type=int, default=128)
    parser.add_argument('--output_dim',     type=int, default=64)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"\n[DATA] Loading synthetic dataset  condition={args.condition}")
    data_dir = project_root / 'data' / 'synthetic'
    loader   = SyntheticInductiveLoader(data_dir=str(data_dir),
                                        condition=args.condition)
    data = loader.load()

    for k, v in data.stats.items():
        print(f"  {k}: {v}")

    trainer = WBTBGRLSyntheticTrainer(
        data           = data,
        model_variant  = args.variant,
        hidden_dim     = args.hidden_dim,
        output_dim     = args.output_dim,
    )

    trainer.train_encoder(epochs=args.epochs_encoder)
    decoder, _, _ = trainer.train_decoder(epochs=args.epochs_decoder)
    results = trainer.evaluate_test(decoder)

    # Save JSON
    ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'model':      f'WBT-BGRL ({args.variant})',
        'dataset':    'synthetic',
        'condition':  args.condition,
        'variant':    args.variant,
        'seed':       args.seed,
        'hidden_dim': args.hidden_dim,
        'output_dim': args.output_dim,
        'results':    results,
        'timestamp':  ts,
    }

    out_dir  = project_root / 'results' / 'synthetic_wbtbgrl'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'synthetic_WBTBGRL_{args.condition}_{args.variant}_{ts}.json'

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"\n[SAVED] {out_file}")


if __name__ == '__main__':
    raise SystemExit('Legacy trainer is not valid for revised tables. Use python revised/run.py; see README_REVISION.md.')
    main()
