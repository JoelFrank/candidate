"""
E-commerce Inductive-Temporal Data Loader

Dataset: E-commerce transactions (Customer-Product bipartite graph)
Features:
  - CUSTOMER nodes: scaled_mean_spend (numeric), country_code (embedding)
  - PRODUCT nodes: scaled_avg_price (numeric), popularity (embedding)

Inductive-Temporal Protocol:
  1. Chronological split: 80% PAST / 10% PRESENT / 10% FUTURE
  2. Vocabularies (Country, Popularity) built ONLY from PAST
  3. Features computed with temporal restrictions:
     - PRESENT features use only PAST history
     - FUTURE features use PAST + PRESENT history
  4. Negative sampling excludes ALL real edges (train + val + test)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


@dataclass
class EcommerceInductiveData:
    """Container for E-commerce inductive-temporal data splits"""
    
    # TRAIN (PAST) - 80%
    train_customer_ids: np.ndarray  # Customer IDs in train
    train_product_ids: np.ndarray   # Product IDs in train
    train_num_customers: int
    train_num_products: int
    train_edges: torch.Tensor  # [2, num_edges] U indices [0, num_U), V indices [num_U, num_U+num_V)
    train_edge_weights: torch.Tensor  # [num_edges] Edge weights (transaction frequencies)
    train_customer_features: torch.Tensor  # [num_U, customer_feat_dim]
    train_product_features: torch.Tensor   # [num_V, product_feat_dim]
    
    # VAL (PRESENT) - 10%
    val_customer_ids: np.ndarray
    val_product_ids: np.ndarray
    val_num_customers: int
    val_num_products: int
    val_edges: torch.Tensor  # Edges to predict (positive examples)
    val_edge_weights: torch.Tensor  # [num_edges] Edge weights (transaction frequencies from VAL split)
    val_customer_features: torch.Tensor  # Features using PAST history only
    val_product_features: torch.Tensor
    
    # TEST (FUTURE) - 10%
    test_customer_ids: np.ndarray
    test_product_ids: np.ndarray
    test_num_customers: int
    test_num_products: int
    test_edges: torch.Tensor  # Edges to predict (positive examples)
    test_edge_weights: torch.Tensor  # [num_edges] Edge weights (transaction frequencies from TEST split)
    test_customer_features: torch.Tensor  # Features using PAST + PRESENT history
    test_product_features: torch.Tensor
    
    # Vocabularies (for embeddings) - built only from TRAIN
    country_vocab: Dict[str, int]  # Country -> index mapping
    popularity_vocab: Dict[int, int]  # Popularity value -> index mapping
    
    # ID mappings
    customer_id_to_idx: Dict[float, int]  # Original CustomerID -> continuous index
    product_id_to_idx: Dict[str, int]     # Original StockCode -> continuous index
    
    # Statistics
    customer_feat_dim: int  # 1 (scaled_mean_spend) + country_embed_dim
    product_feat_dim: int   # 1 (scaled_avg_price) + popularity_embed_dim
    
    # Embedding dimensions
    country_embed_dim: int
    popularity_embed_dim: int


class EcommerceInductiveLoader:
    """
    Loads E-commerce data with strict inductive-temporal protocol.
    
    Features:
      - CUSTOMERS: scaled_mean_spend (log1p + MinMax), country_code (embedding)
      - PRODUCTS: scaled_avg_price (log1p + MinMax), popularity (embedding)
    
    Temporal Guarantees:
      - Vocabularies built only from PAST (train)
      - PRESENT features use only PAST history
      - FUTURE features use PAST + PRESENT history
    """
    
    def __init__(
        self,
        data_path: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        country_embed_dim: int = 8,
        popularity_embed_dim: int = 8,
        verbose: bool = True
    ):
        self.data_path = Path(data_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.country_embed_dim = country_embed_dim
        self.popularity_embed_dim = popularity_embed_dim
        self.verbose = verbose
        
        # Token for unknown values
        self.UNK_COUNTRY = "<UNK_COUNTRY>"
        self.UNK_POPULARITY = "<UNK_POPULARITY>"
    
    def load(self) -> EcommerceInductiveData:
        """Load and process E-commerce data with inductive-temporal protocol"""
        
        if self.verbose:
            print("\n" + "="*80)
            print("E-COMMERCE INDUCTIVE-TEMPORAL DATA LOADER")
            print("="*80)
        
        # Phase 1: Load and chronologically split
        df = self._load_and_clean()
        train_df, val_df, test_df = self._chronological_split(df)
        
        if self.verbose:
            print(f"\n[SPLITS]")
            print(f"  PAST (train):    {len(train_df):,} transactions ({self.train_ratio*100:.0f}%)")
            print(f"  PRESENT (val):   {len(val_df):,} transactions ({self.val_ratio*100:.0f}%)")
            print(f"  FUTURE (test):   {len(test_df):,} transactions ({self.test_ratio*100:.0f}%)")
        
        # Phase 2: Create vocabularies ONLY from TRAIN
        country_vocab = self._build_country_vocab(train_df)
        popularity_vocab = self._build_popularity_vocab(train_df)
        
        # Phase 3: Create ID mappings (from TRAIN only - for bipartite structure)
        customer_id_to_idx, product_id_to_idx = self._create_id_mappings(train_df)
        
        if self.verbose:
            print(f"\n[VOCABULARY - From TRAIN only]")
            print(f"  Countries: {len(country_vocab)} unique")
            print(f"  Popularity bins: {len(popularity_vocab)} unique")
            print(f"\n[ID MAPPINGS]")
            print(f"  Customers: {len(customer_id_to_idx)} unique")
            print(f"  Products: {len(product_id_to_idx)} unique")
        
        # Phase 4: Generate snapshots with temporal restrictions
        
        # TRAIN: Features from TRAIN history
        train_data = self._generate_train_snapshot(
            train_df, customer_id_to_idx, product_id_to_idx,
            country_vocab, popularity_vocab
        )
        
        # VAL: Features from TRAIN history only (not val!)
        val_data = self._generate_val_snapshot(
            train_df, val_df, train_data, customer_id_to_idx, product_id_to_idx,
            country_vocab, popularity_vocab
        )
        
        # TEST: Features from TRAIN + VAL history
        test_data = self._generate_test_snapshot(
            train_df, val_df, test_df, train_data, customer_id_to_idx, product_id_to_idx,
            country_vocab, popularity_vocab
        )
        
        return EcommerceInductiveData(
            # Train
            train_customer_ids=train_data['customer_ids'],
            train_product_ids=train_data['product_ids'],
            train_num_customers=train_data['num_customers'],
            train_num_products=train_data['num_products'],
            train_edges=train_data['edges'],
            train_edge_weights=train_data['edge_weights'],
            train_customer_features=train_data['customer_features'],
            train_product_features=train_data['product_features'],
            
            # Val
            val_customer_ids=val_data['customer_ids'],
            val_product_ids=val_data['product_ids'],
            val_num_customers=val_data['num_customers'],
            val_num_products=val_data['num_products'],
            val_edges=val_data['edges'],
            val_edge_weights=val_data['edge_weights'],
            val_customer_features=val_data['customer_features'],
            val_product_features=val_data['product_features'],
            
            # Test
            test_customer_ids=test_data['customer_ids'],
            test_product_ids=test_data['product_ids'],
            test_num_customers=test_data['num_customers'],
            test_num_products=test_data['num_products'],
            test_edges=test_data['edges'],
            test_edge_weights=test_data['edge_weights'],
            test_customer_features=test_data['customer_features'],
            test_product_features=test_data['product_features'],
            
            # Vocabularies
            country_vocab=country_vocab,
            popularity_vocab=popularity_vocab,
            
            # ID mappings
            customer_id_to_idx=customer_id_to_idx,
            product_id_to_idx=product_id_to_idx,
            
            # Dimensions
            customer_feat_dim=1 + self.country_embed_dim,
            product_feat_dim=1 + self.popularity_embed_dim,
            country_embed_dim=self.country_embed_dim,
            popularity_embed_dim=self.popularity_embed_dim
        )
    
    def _load_and_clean(self) -> pd.DataFrame:
        """Load CSV and perform basic cleaning"""
        df = pd.read_csv(self.data_path)
        
        # Keep only rows with valid CustomerID and StockCode
        df = df.dropna(subset=['CustomerID', 'StockCode'])
        
        # Convert StockCode to string to avoid type comparison issues
        df['StockCode'] = df['StockCode'].astype(str)
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M')
        
        # Create TotalPrice
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        # Keep only positive prices
        df = df[df['TotalPrice'] > 0].copy()
        
        # Sort chronologically (CRITICAL!)
        df = df.sort_values('InvoiceDate').reset_index(drop=True)
        
        if self.verbose:
            print(f"\n[DATA LOADED]")
            print(f"  Path: {self.data_path}")
            print(f"  Transactions: {len(df):,}")
            print(f"  Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
            print(f"  Unique customers: {df['CustomerID'].nunique():,}")
            print(f"  Unique products: {df['StockCode'].nunique():,}")
        
        return df
    
    def _chronological_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split chronologically: 80% train / 10% val / 10% test"""
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df
    
    def _build_country_vocab(self, train_df: pd.DataFrame) -> Dict[str, int]:
        """Build country vocabulary from TRAIN only"""
        countries = train_df['Country'].unique()
        vocab = {self.UNK_COUNTRY: 0}
        for i, country in enumerate(sorted(countries), start=1):
            vocab[country] = i
        return vocab
    
    def _build_popularity_vocab(self, train_df: pd.DataFrame) -> Dict[int, int]:
        """Build popularity vocabulary from TRAIN only"""
        # Popularity = number of unique customers per product
        popularity = train_df.groupby('StockCode')['CustomerID'].nunique().values
        unique_pops = sorted(set(popularity))
        
        vocab = {self.UNK_POPULARITY: 0}
        for i, pop in enumerate(unique_pops, start=1):
            vocab[pop] = i
        
        return vocab
    
    def _create_id_mappings(
        self, train_df: pd.DataFrame
    ) -> Tuple[Dict[float, int], Dict[str, int]]:
        """Create continuous ID mappings for bipartite structure"""
        # Customers: 0 to N-1
        unique_customers = sorted(train_df['CustomerID'].unique())
        customer_id_to_idx = {cid: i for i, cid in enumerate(unique_customers)}
        
        # Products: N to N+M-1 (offset to avoid overlap)
        unique_products = sorted(train_df['StockCode'].unique())
        offset = len(customer_id_to_idx)
        product_id_to_idx = {pid: i + offset for i, pid in enumerate(unique_products)}
        
        return customer_id_to_idx, product_id_to_idx
    
    def _compute_customer_features(
        self,
        customer_ids: np.ndarray,
        history_df: pd.DataFrame,
        customer_id_to_idx: Dict[float, int],
        country_vocab: Dict[str, int]
    ) -> torch.Tensor:
        """
        Compute customer features: [scaled_mean_spend, country_code_idx]
        
        Args:
            customer_ids: Array of customer IDs to compute features for
            history_df: Historical data to compute features from
            customer_id_to_idx: Customer ID to index mapping
            country_vocab: Country vocabulary
        """
        features = []
        
        for cid in customer_ids:
            # Get customer's transactions in history
            cust_txns = history_df[history_df['CustomerID'] == cid]
            
            if len(cust_txns) > 0:
                # scaled_mean_spend: log1p + will be normalized later
                mean_spend = cust_txns['TotalPrice'].mean()
                scaled_spend = np.log1p(mean_spend)
                
                # country_code: get most common country
                country = cust_txns['Country'].mode()[0]
                country_idx = country_vocab.get(country, 0)  # 0 = UNK
            else:
                # New customer (shouldn't happen in train, but can in val/test)
                scaled_spend = 0.0
                country_idx = 0  # UNK
            
            features.append([scaled_spend, country_idx])
        
        features = np.array(features, dtype=np.float32)
        
        # Normalize scaled_mean_spend (column 0)
        if len(features) > 0:
            scaler = MinMaxScaler()
            features[:, 0:1] = scaler.fit_transform(features[:, 0:1])
        
        return torch.from_numpy(features)
    
    def _compute_product_features(
        self,
        product_ids: np.ndarray,
        history_df: pd.DataFrame,
        product_id_to_idx: Dict[str, int],
        popularity_vocab: Dict[int, int]
    ) -> torch.Tensor:
        """
        Compute product features: [scaled_avg_price, popularity_idx]
        
        Args:
            product_ids: Array of product IDs (StockCodes) to compute features for
            history_df: Historical data to compute features from
            product_id_to_idx: Product ID to index mapping
            popularity_vocab: Popularity vocabulary
        """
        features = []
        
        for pid in product_ids:
            # Get product's transactions in history
            prod_txns = history_df[history_df['StockCode'] == pid]
            
            if len(prod_txns) > 0:
                # scaled_avg_price: log1p + will be normalized later
                avg_price = prod_txns['UnitPrice'].mean()
                scaled_price = np.log1p(avg_price)
                
                # popularity: number of unique customers
                popularity = prod_txns['CustomerID'].nunique()
                popularity_idx = popularity_vocab.get(popularity, 0)  # 0 = UNK
            else:
                # New product (shouldn't happen in train, but can in val/test)
                scaled_price = 0.0
                popularity_idx = 0  # UNK
            
            features.append([scaled_price, popularity_idx])
        
        features = np.array(features, dtype=np.float32)
        
        # Normalize scaled_avg_price (column 0)
        if len(features) > 0:
            scaler = MinMaxScaler()
            features[:, 0:1] = scaler.fit_transform(features[:, 0:1])
        
        return torch.from_numpy(features)
    
    def _generate_train_snapshot(
        self,
        train_df: pd.DataFrame,
        customer_id_to_idx: Dict[float, int],
        product_id_to_idx: Dict[str, int],
        country_vocab: Dict[str, int],
        popularity_vocab: Dict[int, int]
    ) -> Dict:
        """Generate train snapshot: features and edges"""
        
        # Get unique IDs from train
        train_customer_ids = np.array(sorted(train_df['CustomerID'].unique()))
        train_product_ids = np.array(sorted(train_df['StockCode'].unique()))
        
        # Compute features using train history
        customer_features = self._compute_customer_features(
            train_customer_ids, train_df, customer_id_to_idx, country_vocab
        )
        product_features = self._compute_product_features(
            train_product_ids, train_df, product_id_to_idx, popularity_vocab
        )
        
        # Build edges (unique customer-product pairs) and compute weights (transaction frequencies)
        edges_df = train_df[['CustomerID', 'StockCode']].value_counts().reset_index(name='frequency')
        
        # Map to continuous indices
        u_indices = edges_df['CustomerID'].map(customer_id_to_idx).values
        v_indices = edges_df['StockCode'].map(product_id_to_idx).values
        edge_weights = edges_df['frequency'].values
        
        edges = torch.tensor(np.array([u_indices, v_indices]), dtype=torch.long)
        edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float)
        
        if self.verbose:
            print(f"\n[TRAIN SNAPSHOT - PAST]")
            print(f"  Customers: {len(train_customer_ids)}")
            print(f"  Products: {len(train_product_ids)}")
            print(f"  Edges: {edges.shape[1]}")
            print(f"  Edge weight stats: min={edge_weights.min()}, max={edge_weights.max()}, mean={edge_weights.mean():.2f}")
            print(f"  Customer features: {customer_features.shape}")
            print(f"  Product features: {product_features.shape}")
        
        return {
            'customer_ids': train_customer_ids,
            'product_ids': train_product_ids,
            'num_customers': len(train_customer_ids),
            'num_products': len(train_product_ids),
            'edges': edges,
            'edge_weights': edge_weights_tensor,
            'customer_features': customer_features,
            'product_features': product_features
        }
    
    def _generate_val_snapshot(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        train_data: Dict,
        customer_id_to_idx: Dict[float, int],
        product_id_to_idx: Dict[str, int],
        country_vocab: Dict[str, int],
        popularity_vocab: Dict[int, int]
    ) -> Dict:
        """Generate val snapshot: features from TRAIN history only"""
        
        # Get IDs that appear in train OR val
        val_customer_ids = np.array(sorted(
            set(train_df['CustomerID'].unique()) | set(val_df['CustomerID'].unique())
        ))
        val_product_ids = np.array(sorted(
            set(train_df['StockCode'].unique()) | set(val_df['StockCode'].unique())
        ))
        
        # CRITICAL: Features computed using TRAIN history only
        customer_features = self._compute_customer_features(
            val_customer_ids, train_df, customer_id_to_idx, country_vocab
        )
        product_features = self._compute_product_features(
            val_product_ids, train_df, product_id_to_idx, popularity_vocab
        )
        
        # Edges to predict: only from VAL
        edges_df = val_df[['CustomerID', 'StockCode']].drop_duplicates()
        
        # Filter to keep only edges where both nodes exist in our vocabulary
        edges_df = edges_df[
            edges_df['CustomerID'].isin(customer_id_to_idx) &
            edges_df['StockCode'].isin(product_id_to_idx)
        ]
        
        # Create temporary local mappings for val space
        val_customer_to_local = {cid: i for i, cid in enumerate(val_customer_ids)}
        val_product_to_local = {pid: i for i, pid in enumerate(val_product_ids)}
        
        u_indices = edges_df['CustomerID'].map(val_customer_to_local).values
        v_offset = len(val_customer_ids)
        v_indices = edges_df['StockCode'].map(val_product_to_local).values + v_offset
        
        edges = torch.tensor(np.array([u_indices, v_indices]), dtype=torch.long)
        
        # Compute edge frequencies from val_df (split's own data, not TRAIN)
        val_edge_freq = val_df[['CustomerID', 'StockCode']].value_counts().to_dict()
        train_edge_freq = train_df[['CustomerID', 'StockCode']].value_counts().to_dict()
        
        # Map edge weights for val edges
        edge_weights_list = []
        for idx, row in edges_df.iterrows():
            cust_id = row['CustomerID']
            prod_id = row['StockCode']
            weight = float(val_edge_freq.get((cust_id, prod_id), 1.0))
            edge_weights_list.append(weight)
        
        edge_weights = torch.tensor(edge_weights_list, dtype=torch.float)
        # Count new edges: pairs not seen in TRAIN
        num_new_edges = sum(
            1 for _, row in edges_df.iterrows()
            if (row['CustomerID'], row['StockCode']) not in train_edge_freq
        )
        
        if self.verbose:
            print(f"\n[VAL SNAPSHOT - PRESENT]")
            print(f"  Customers: {len(val_customer_ids)} (features from PAST only)")
            print(f"  Products: {len(val_product_ids)} (features from PAST only)")
            print(f"  Edges: {edges.shape[1]} ({num_new_edges} nuevos pares no vistos en TRAIN)")
            print(f"  Edge weights: min={edge_weights.min():.1f}, max={edge_weights.max():.1f}, mean={edge_weights.mean():.2f}")
            print(f"  Customer features: {customer_features.shape}")
            print(f"  Product features: {product_features.shape}")
        
        return {
            'customer_ids': val_customer_ids,
            'product_ids': val_product_ids,
            'num_customers': len(val_customer_ids),
            'num_products': len(val_product_ids),
            'edges': edges,
            'edge_weights': edge_weights,  # ✅ NEW
            'customer_features': customer_features,
            'product_features': product_features
        }
    
    def _generate_test_snapshot(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_data: Dict,
        customer_id_to_idx: Dict[float, int],
        product_id_to_idx: Dict[str, int],
        country_vocab: Dict[str, int],
        popularity_vocab: Dict[int, int]
    ) -> Dict:
        """Generate test snapshot: features from TRAIN + VAL history"""
        
        # Get IDs that appear in train OR val OR test
        test_customer_ids = np.array(sorted(
            set(train_df['CustomerID'].unique()) |
            set(val_df['CustomerID'].unique()) |
            set(test_df['CustomerID'].unique())
        ))
        test_product_ids = np.array(sorted(
            set(train_df['StockCode'].unique()) |
            set(val_df['StockCode'].unique()) |
            set(test_df['StockCode'].unique())
        ))
        
        # Features computed using TRAIN + VAL history
        history_df = pd.concat([train_df, val_df], ignore_index=True)
        
        customer_features = self._compute_customer_features(
            test_customer_ids, history_df, customer_id_to_idx, country_vocab
        )
        product_features = self._compute_product_features(
            test_product_ids, history_df, product_id_to_idx, popularity_vocab
        )
        
        # Edges to predict: only from TEST
        # KEEP drop_duplicates() - eliminates repeated (Customer, Product) pairs
        # But DO NOT filter by vocabulary - evaluate on ALL unique edges (including new nodes)
        edges_df = test_df[['CustomerID', 'StockCode']].drop_duplicates()
        
        # Create temporary local mappings for test space
        test_customer_to_local = {cid: i for i, cid in enumerate(test_customer_ids)}
        test_product_to_local = {pid: i for i, pid in enumerate(test_product_ids)}
        
        u_indices = edges_df['CustomerID'].map(test_customer_to_local).values
        v_offset = len(test_customer_ids)
        v_indices = edges_df['StockCode'].map(test_product_to_local).values + v_offset
        
        edges = torch.tensor(np.array([u_indices, v_indices]), dtype=torch.long)
        
        # Compute edge frequencies from test_df (split's own data, not TRAIN)
        test_edge_freq = test_df[['CustomerID', 'StockCode']].value_counts().to_dict()
        train_val_edge_freq = pd.concat([train_df, val_df])[['CustomerID', 'StockCode']].value_counts().to_dict()
        
        # Map edge weights for test edges
        edge_weights_list = []
        for idx, row in edges_df.iterrows():
            cust_id = row['CustomerID']
            prod_id = row['StockCode']
            weight = float(test_edge_freq.get((cust_id, prod_id), 1.0))
            edge_weights_list.append(weight)
        
        edge_weights = torch.tensor(edge_weights_list, dtype=torch.float)
        # Count new edges: pairs not seen in TRAIN or VAL
        num_new_edges = sum(
            1 for _, row in edges_df.iterrows()
            if (row['CustomerID'], row['StockCode']) not in train_val_edge_freq
        )
        
        if self.verbose:
            print(f"\n[TEST SNAPSHOT - FUTURE]")
            print(f"  Customers: {len(test_customer_ids)} (features from PAST + PRESENT)")
            print(f"  Products: {len(test_product_ids)} (features from PAST + PRESENT)")
            print(f"  Edges: {edges.shape[1]} ({num_new_edges} nuevos pares no vistos en TRAIN+VAL)")
            print(f"  Edge weights: min={edge_weights.min():.1f}, max={edge_weights.max():.1f}, mean={edge_weights.mean():.2f}")
            print(f"  Customer features: {customer_features.shape}")
            print(f"  Product features: {product_features.shape}")
        
        return {
            'customer_ids': test_customer_ids,
            'product_ids': test_product_ids,
            'num_customers': len(test_customer_ids),
            'num_products': len(test_product_ids),
            'edges': edges,
            'edge_weights': edge_weights,  # ✅ NEW
            'customer_features': customer_features,
            'product_features': product_features
        }
