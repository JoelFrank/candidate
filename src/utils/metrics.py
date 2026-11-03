"""
Metrics for evaluating link prediction performance.
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score, accuracy_score
)
from typing import Dict, List, Tuple, Optional, Union
import torch.nn.functional as F


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computa métricas completas para evaluación de predicción de enlaces
    
    Args:
        labels: Etiquetas verdaderas (1 para positivos, 0 para negativos)
        scores: Puntuaciones predichas
        threshold: Umbral para clasificación binaria
        
    Returns:
        Diccionario con todas las métricas
    """
    # [FIX] CORRECCIÓN: Verificar y limpiar NaN en scores
    if np.isnan(scores).any():
        print("[U+26A0][U+FE0F] Detectados NaN en scores. Reemplazando con 0.5...")
        scores = np.nan_to_num(scores, nan=0.5)  # Reemplazar NaN con valor neutro
    
    # Verificar casos extremos
    if np.all(scores == scores[0]):  # Todos los scores iguales
        print("[U+26A0][U+FE0F] Todos los scores son iguales. Agregando ruido mínimo...")
        scores = scores + np.random.normal(0, 1e-8, len(scores))
    
    try:
        # ROC-AUC
        roc_auc = roc_auc_score(labels, scores)
    except ValueError as e:
        print(f"[U+26A0][U+FE0F] Error en ROC-AUC: {e}. Usando 0.5 como fallback.")
        roc_auc = 0.5
    
    try:
        # Average Precision (AP)
        average_precision = average_precision_score(labels, scores)
    except ValueError as e:
        print(f"[U+26A0][U+FE0F] Error en Average Precision: {e}. Usando proporción de positivos.")
        average_precision = np.mean(labels)
    
    # Convertir scores a predicciones binarias para métricas de clasificación
    predictions = (scores >= threshold).astype(int)
    
    # Precision, Recall, F1-Score
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # Accuracy
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'roc_auc': roc_auc,
        'average_precision': average_precision,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    }


def compute_link_prediction_metrics(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    k_values: List[int] = [10, 50, 100]
) -> Dict[str, float]:
    """
    Compute comprehensive link prediction metrics.
    
    Args:
        pos_scores: Scores for positive edges [num_pos]
        neg_scores: Scores for negative edges [num_neg]
        k_values: List of k values for Hits@k
        
    Returns:
        Dictionary of metrics
    """
    pos_scores = pos_scores.cpu().numpy()
    neg_scores = neg_scores.cpu().numpy()
    
    # Combine scores and labels
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([
        np.ones(len(pos_scores)),
        np.zeros(len(neg_scores))
    ])
    
    # Usar la nueva función compute_metrics
    metrics = compute_metrics(all_labels, all_scores)
    
    # Agregar métricas de ranking adicionales
    # Hits@k
    for k in k_values:
        hits_k = compute_hits_at_k(pos_scores, neg_scores, k)
        metrics[f'hits@{k}'] = hits_k
    
    # Mean Reciprocal Rank (MRR)
    metrics['mrr'] = compute_mrr(pos_scores, neg_scores)
    
    return metrics


def compute_hits_at_k(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    k: int
) -> float:
    """
    Compute Hits@k metric.
    
    For each positive edge, rank it against all negative edges.
    Hits@k is the fraction of positive edges ranked in top-k.
    """
    hits = 0
    total = len(pos_scores)
    
    for pos_score in pos_scores:
        # Count how many negative scores are higher
        rank = np.sum(neg_scores >= pos_score) + 1
        if rank <= k:
            hits += 1
    
    return hits / total


def compute_mrr(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    """
    reciprocal_ranks = []
    
    for pos_score in pos_scores:
        # Count how many negative scores are higher
        rank = np.sum(neg_scores >= pos_score) + 1
        reciprocal_ranks.append(1.0 / rank)
    
    return np.mean(reciprocal_ranks)


def compute_ranking_metrics(
    embeddings_u: torch.Tensor,
    embeddings_v: torch.Tensor,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
    scoring_function: str = 'dot_product'
) -> Dict[str, float]:
    """
    Compute ranking metrics from embeddings.
    
    Args:
        embeddings_u: Embeddings for U nodes [num_u, dim]
        embeddings_v: Embeddings for V nodes [num_v, dim]
        pos_edges: Positive edges [2, num_pos]
        neg_edges: Negative edges [2, num_neg]
        scoring_function: Function to compute edge scores
        
    Returns:
        Dictionary of metrics
    """
    # Compute scores for positive and negative edges
    pos_scores = compute_edge_scores(
        embeddings_u, embeddings_v, pos_edges, scoring_function
    )
    neg_scores = compute_edge_scores(
        embeddings_u, embeddings_v, neg_edges, scoring_function
    )
    
    return compute_link_prediction_metrics(pos_scores, neg_scores)


def compute_edge_scores(
    embeddings_u: torch.Tensor,
    embeddings_v: torch.Tensor,
    edges: torch.Tensor,
    scoring_function: str = 'dot_product'
) -> torch.Tensor:
    """
    Compute edge scores using embeddings.
    
    Args:
        embeddings_u: Embeddings for U nodes [num_u, dim]
        embeddings_v: Embeddings for V nodes [num_v, dim]
        edges: Edge indices [2, num_edges]
        scoring_function: Scoring function type
        
    Returns:
        Edge scores [num_edges]
    """
    # Get embeddings for edge endpoints
    u_embs = embeddings_u[edges[0]]  # [num_edges, dim]
    
    # Adjust v indices to local indexing
    v_indices_local = edges[1] - embeddings_u.size(0)
    v_embs = embeddings_v[v_indices_local]  # [num_edges, dim]
    
    if scoring_function == 'dot_product':
        scores = torch.sum(u_embs * v_embs, dim=1)
    elif scoring_function == 'cosine':
        u_norm = F.normalize(u_embs, p=2, dim=1)
        v_norm = F.normalize(v_embs, p=2, dim=1)
        scores = torch.sum(u_norm * v_norm, dim=1)
    elif scoring_function == 'l2_distance':
        scores = -torch.norm(u_embs - v_embs, p=2, dim=1)
    elif scoring_function == 'hadamard':
        scores = torch.sum(u_embs * v_embs, dim=1)  # Same as dot product for single layer
    else:
        raise ValueError(f"Unknown scoring function: {scoring_function}")
    
    return scores


def evaluate_node_classification(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate node classification performance.
    
    Args:
        embeddings: Node embeddings [num_nodes, dim]
        labels: Node labels [num_nodes]
        train_mask: Training mask [num_nodes]
        test_mask: Test mask [num_nodes]
        
    Returns:
        Classification metrics
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    
    # Extract training and test data
    X_train = embeddings[train_mask].cpu().numpy()
    y_train = labels[train_mask].cpu().numpy()
    X_test = embeddings[test_mask].cpu().numpy()
    y_test = labels[test_mask].cpu().numpy()
    
    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_micro': f1_score(y_test, y_pred, average='micro')
    }
    
    return metrics


def compute_embedding_quality_metrics(
    embeddings_u: torch.Tensor,
    embeddings_v: torch.Tensor,
    edge_index: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics to assess embedding quality.
    
    Args:
        embeddings_u: Embeddings for U nodes
        embeddings_v: Embeddings for V nodes  
        edge_index: Edge indices
        
    Returns:
        Quality metrics
    """
    metrics = {}
    
    # Compute embedding norms
    u_norms = torch.norm(embeddings_u, p=2, dim=1)
    v_norms = torch.norm(embeddings_v, p=2, dim=1)
    
    metrics['mean_u_norm'] = u_norms.mean().item()
    metrics['mean_v_norm'] = v_norms.mean().item()
    metrics['std_u_norm'] = u_norms.std().item()
    metrics['std_v_norm'] = v_norms.std().item()
    
    # Compute cosine similarities for connected pairs
    u_indices = edge_index[0]
    v_indices = edge_index[1] - embeddings_u.size(0)  # Adjust to local indexing
    
    u_embs = embeddings_u[u_indices]
    v_embs = embeddings_v[v_indices]
    
    u_norm = F.normalize(u_embs, p=2, dim=1)
    v_norm = F.normalize(v_embs, p=2, dim=1)
    
    cosine_sims = torch.sum(u_norm * v_norm, dim=1)
    
    metrics['mean_pos_cosine_sim'] = cosine_sims.mean().item()
    metrics['std_pos_cosine_sim'] = cosine_sims.std().item()
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics."""
    print(f"\n=== {title} ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print()


class MetricsLogger:
    """Logger for tracking metrics during training."""
    
    def __init__(self):
        self.history = {}
    
    def log(self, metrics: Dict[str, float], epoch: int):
        """Log metrics for an epoch."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((epoch, value))
    
    def get_best(self, metric: str, mode: str = 'max') -> Tuple[int, float]:
        """Get best value for a metric."""
        if metric not in self.history:
            return -1, 0.0
        
        values = self.history[metric]
        if mode == 'max':
            best_epoch, best_value = max(values, key=lambda x: x[1])
        else:
            best_epoch, best_value = min(values, key=lambda x: x[1])
        
        return best_epoch, best_value
    
    def get_latest(self, metric: str) -> float:
        """Get latest value for a metric."""
        if metric not in self.history or len(self.history[metric]) == 0:
            return 0.0
        return self.history[metric][-1][1]
