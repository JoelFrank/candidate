"""
Visualization utilities for bipartite graphs and embeddings.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Optional, Tuple, List, Dict
import networkx as nx


def plot_embeddings(
    embeddings_u: torch.Tensor,
    embeddings_v: torch.Tensor,
    method: str = 'tsne',
    save_path: Optional[str] = None,
    title: str = "Bipartite Graph Embeddings"
):
    """
    Plot 2D visualization of node embeddings.
    
    Args:
        embeddings_u: Embeddings for U nodes [num_u, dim]
        embeddings_v: Embeddings for V nodes [num_v, dim]
        method: Dimensionality reduction method ('tsne' or 'pca')
        save_path: Path to save the plot
        title: Plot title
    """
    # Convert to numpy
    emb_u = embeddings_u.detach().cpu().numpy()
    emb_v = embeddings_v.detach().cpu().numpy()
    
    # Combine embeddings
    all_embeddings = np.vstack([emb_u, emb_v])
    
    # Create labels
    labels = ['U'] * len(emb_u) + ['V'] * len(emb_v)
    
    # Apply dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    reduced_embeddings = reducer.fit_transform(all_embeddings)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot U nodes
    u_points = reduced_embeddings[:len(emb_u)]
    plt.scatter(u_points[:, 0], u_points[:, 1], 
               c='blue', alpha=0.6, label='U nodes', s=50)
    
    # Plot V nodes
    v_points = reduced_embeddings[len(emb_u):]
    plt.scatter(v_points[:, 0], v_points[:, 1], 
               c='red', alpha=0.6, label='V nodes', s=50)
    
    plt.title(f"{title} ({method.upper()})")
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(
    metrics_history: Dict[str, List[Tuple[int, float]]],
    save_path: Optional[str] = None
):
    """
    Plot training curves for metrics.
    
    Args:
        metrics_history: Dictionary of metric histories
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metric_names = list(metrics_history.keys())[:4]  # Plot up to 4 metrics
    
    for i, metric_name in enumerate(metric_names):
        if i >= 4:
            break
            
        history = metrics_history[metric_name]
        epochs = [x[0] for x in history]
        values = [x[1] for x in history]
        
        axes[i].plot(epochs, values, 'b-', linewidth=2)
        axes[i].set_title(f"{metric_name}")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric_name)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(metric_names), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_bipartite_graph(
    edge_index: torch.Tensor,
    num_nodes_u: int,
    num_nodes_v: int,
    max_nodes: int = 100,
    save_path: Optional[str] = None,
    title: str = "Bipartite Graph"
):
    """
    Plot bipartite graph structure.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes_u: Number of U nodes
        num_nodes_v: Number of V nodes
        max_nodes: Maximum number of nodes to display
        save_path: Path to save the plot
        title: Plot title
    """
    # Subsample if graph is too large
    if num_nodes_u + num_nodes_v > max_nodes:
        ratio = max_nodes / (num_nodes_u + num_nodes_v)
        sample_u = int(num_nodes_u * ratio)
        sample_v = int(num_nodes_v * ratio)
        
        # Sample nodes
        u_nodes = torch.randperm(num_nodes_u)[:sample_u]
        v_nodes = torch.randperm(num_nodes_v)[:sample_v] + num_nodes_u
        
        # Filter edges
        edge_mask = (
            torch.isin(edge_index[0], u_nodes) & 
            torch.isin(edge_index[1], v_nodes)
        )
        edge_index = edge_index[:, edge_mask]
        
        num_nodes_u = sample_u
        num_nodes_v = sample_v
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes with bipartite attribute
    for i in range(num_nodes_u):
        G.add_node(i, bipartite=0)
    for i in range(num_nodes_u, num_nodes_u + num_nodes_v):
        G.add_node(i, bipartite=1)
    
    # Add edges
    edges = edge_index.T.numpy()
    G.add_edges_from(edges)
    
    # Create bipartite layout
    pos = {}
    
    # Position U nodes on the left
    for i, node in enumerate(range(num_nodes_u)):
        pos[node] = (0, i / max(1, num_nodes_u - 1))
    
    # Position V nodes on the right
    for i, node in enumerate(range(num_nodes_u, num_nodes_u + num_nodes_v)):
        pos[node] = (1, i / max(1, num_nodes_v - 1))
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    u_nodes = list(range(num_nodes_u))
    v_nodes = list(range(num_nodes_u, num_nodes_u + num_nodes_v))
    
    nx.draw_networkx_nodes(G, pos, nodelist=u_nodes, 
                          node_color='lightblue', node_size=100, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=v_nodes, 
                          node_color='lightcoral', node_size=100, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    
    plt.title(f"{title} (Showing {num_nodes_u + num_nodes_v} nodes)")
    plt.xlabel("U nodes [U+2190] -> V nodes")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_degree_distribution(
    edge_index: torch.Tensor,
    num_nodes_u: int,
    num_nodes_v: int,
    save_path: Optional[str] = None
):
    """
    Plot degree distribution for bipartite graph.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes_u: Number of U nodes
        num_nodes_v: Number of V nodes
        save_path: Path to save the plot
    """
    # Compute degrees
    degrees_u = torch.bincount(edge_index[0], minlength=num_nodes_u).numpy()
    degrees_v = torch.bincount(edge_index[1] - num_nodes_u, minlength=num_nodes_v).numpy()
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot U node degrees
    ax1.hist(degrees_u, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title("Degree Distribution - U Nodes")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)
    
    # Plot V node degrees
    ax2.hist(degrees_v, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax2.set_title("Degree Distribution - V Nodes")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_score_distributions(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Link Prediction Score Distributions"
):
    """
    Plot distributions of positive and negative link scores.
    
    Args:
        pos_scores: Scores for positive edges
        neg_scores: Scores for negative edges
        save_path: Path to save the plot
        title: Plot title
    """
    pos_scores = pos_scores.detach().cpu().numpy()
    neg_scores = neg_scores.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(neg_scores, bins=50, alpha=0.7, label='Negative edges', 
             color='red', density=True)
    plt.hist(pos_scores, bins=50, alpha=0.7, label='Positive edges', 
             color='blue', density=True)
    
    # Add vertical lines for means
    plt.axvline(np.mean(neg_scores), color='red', linestyle='--', 
                label=f'Neg mean: {np.mean(neg_scores):.3f}')
    plt.axvline(np.mean(pos_scores), color='blue', linestyle='--', 
                label=f'Pos mean: {np.mean(pos_scores):.3f}')
    
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_results_summary_plot(
    results: Dict[str, Dict[str, float]],
    metric: str = 'hits@50',
    save_path: Optional[str] = None
):
    """
    Create summary plot comparing different methods.
    
    Args:
        results: Dictionary of {method_name: {metric: value}}
        metric: Metric to plot
        save_path: Path to save the plot
    """
    methods = list(results.keys())
    values = [results[method].get(metric, 0) for method in methods]
    
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    bars = plt.bar(methods, values, alpha=0.8, 
                   color=['blue', 'red', 'green', 'orange'][:len(methods)])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f"Method Comparison - {metric}")
    plt.ylabel(metric)
    plt.xlabel("Method")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at best performance
    if values:
        best_value = max(values)
        plt.axhline(y=best_value, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
