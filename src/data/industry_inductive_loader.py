"""
INDUSTRY Inductive Temporal Loader - SIMPLIFIED VERSION

Implements the simplified inductive-temporal pipeline for INDUSTRY data:
- Chronological splitting (80% train, 10% val, 10% test)
- Vocabulary creation from training data only
- Embedding-based node representations
- Global temporal evaluation (no O-O/O-U/U-U breakdown)
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class INDUSTRYVocabulary:
    """Vocabularies for INDUSTRY nodes and attributes."""
    loc_to_idx: Dict[str, int]  # LOCALIZAÇÃO ID -> index
    patrimonio_to_idx: Dict[str, int]  # OBSERVAÇÃO PATRIMÔNIO -> index (V nodes)
    tipo_to_idx: Dict[str, int]  # TIPO DO EQUIPAMENTO -> index (V attribute)
    idx_to_loc: Dict[int, str]
    idx_to_patrimonio: Dict[int, str]
    idx_to_tipo: Dict[int, str]
    unk_loc_idx: int
    unk_patrimonio_idx: int
    unk_tipo_idx: int
    num_locs: int  # Including <UNK>
    num_patrimonios: int  # Including <UNK>
    num_tipos: int  # Including <UNK>


@dataclass
class INDUSTRYInductiveData:
    """Complete inductive data structure for INDUSTRY - Simplified."""
    # Training data (PAST - 80%)
    train_edge_index: torch.LongTensor  # [2, num_train_edges]
    train_edge_weights: torch.FloatTensor  # [num_train_edges] - ✅ NEW: edge frequencies
    train_loc_ids: torch.LongTensor  # [num_train_locs]
    train_patrimonio_ids: torch.LongTensor  # [num_train_patrimonios] (V nodes)
    train_patrimonio_tipo_ids: torch.LongTensor  # [num_train_patrimonios] (TIPO attribute for each V node)
    train_num_locs: int
    train_num_patrimonios: int
    
    # Validation data (PRESENT - 10%)
    val_edge_index: torch.LongTensor  # [2, num_val_edges]
    val_edge_weights: torch.FloatTensor  # [num_val_edges] - edge frequencies from VAL split
    val_loc_ids: torch.LongTensor  # [num_val_locs] (includes <UNK>)
    val_patrimonio_ids: torch.LongTensor  # [num_val_patrimonios] (includes <UNK>)
    val_patrimonio_tipo_ids: torch.LongTensor  # [num_val_patrimonios] (TIPO attribute)
    val_num_locs: int
    val_num_patrimonios: int
    
    # Test data (FUTURE - 10%)
    test_edge_index: torch.LongTensor  # [2, num_test_edges]
    test_edge_weights: torch.FloatTensor  # [num_test_edges] - edge frequencies from TEST split
    test_loc_ids: torch.LongTensor  # [num_test_locs] (includes <UNK>)
    test_patrimonio_ids: torch.LongTensor  # [num_test_patrimonios] (includes <UNK>)
    test_patrimonio_tipo_ids: torch.LongTensor  # [num_test_patrimonios] (TIPO attribute)
    test_num_locs: int
    test_num_patrimonios: int
    
    # Vocabularies
    vocabulary: INDUSTRYVocabulary
    
    # Statistics
    stats: Dict[str, int]


class INDUSTRYInductiveLoader:
    """
    Loads INDUSTRY data following the SIMPLIFIED inductive temporal protocol.
    
    Key principles:
    1. Chronological splitting (80% train / 10% val / 10% test)
    2. Vocabulary built ONLY from training data (PAST)
    3. Embedding layers learn node representations
    4. <UNK> tokens for unseen nodes in val/test (PRESENT/FUTURE)
    5. Global temporal evaluation (no O-O/O-U/U-U breakdown)
    
    Timeline:
    - TRAIN (80%): Past data -> Train encoder self-supervised
    - VAL (10%): Present data -> Train decoder supervised with early stopping
    - TEST (10%): Future data -> Final global evaluation
    """
    
    def __init__(
        self,
        excel_path: str,
        train_ratio: float = 0.8,  # 80% PAST
        val_ratio: float = 0.1,    # 10% PRESENT
        test_ratio: float = 0.1,   # 10% FUTURE
        date_column: str = 'DATA DE ABERTURA',
        loc_column: str = 'LOCALIZAÇÃO',
        patrimonio_column: str = 'OBSERVAÇÃO PATRIMÔNIO',  # V nodes
        tipo_column: str = 'TIPO DO EQUIPAMENTO',  # V attribute
        seed: int = 42
    ):
        self.excel_path = excel_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.date_column = date_column
        self.loc_column = loc_column
        self.patrimonio_column = patrimonio_column
        self.tipo_column = tipo_column
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def load(self) -> INDUSTRYInductiveData:
        """Load and process INDUSTRY data following simplified inductive protocol."""
        print("=" * 80)
        print("INDUSTRY INDUCTIVE TEMPORAL LOADER - SIMPLIFIED")
        print("=" * 80)
        
        # Phase 1: Load and chronologically split
        print("\n[PHASE 1] FASE 1: Carga y División Cronológica...")
        df = self._load_and_prepare_data()
        train_df, val_df, test_df = self._chronological_split(df)
        
        print(f"  [OK] PASADO (Train):   {len(train_df)} interacciones ({self.train_ratio:.0%})")
        print(f"  [OK] PRESENTE (Val):   {len(val_df)} interacciones ({self.val_ratio:.0%})")
        print(f"  [OK] FUTURO (Test):    {len(test_df)} interacciones ({self.test_ratio:.0%})")
        
        # Phase 2: Build vocabularies from training data ONLY
        print("\n[PHASE 2] FASE 2: Construcción de Vocabularios (Solo Datos de Entrenamiento)...")
        vocabulary = self._build_vocabularies(train_df)
        
        print(f"  [OK] LOCALIZAÇÕEs: {vocabulary.num_locs-1} conocidos + 1 <UNK>")
        print(f"  [OK] PATRIMÔNIOS:  {vocabulary.num_patrimonios-1} conocidos + 1 <UNK>")
        print(f"  [OK] TIPO DOs (attr): {vocabulary.num_tipos-1} conocidos + 1 <UNK>")
        
        # Phase 3: Process training data
        print("\n[PHASE 3]  FASE 3: Procesamiento de Datos de Entrenamiento...")
        train_data = self._process_training_data(train_df, vocabulary)
        
        # Phase 4: Process validation data
        print("\n[PHASE 4] FASE 4: Procesamiento de Datos de Validación...")
        val_data = self._process_split_data(val_df, vocabulary, train_data, "Validación")
        
        # Phase 5: Process test data
        print("\n[PHASE 5] FASE 5: Procesamiento de Datos de Prueba...")
        test_data = self._process_split_data(test_df, vocabulary, train_data, "Prueba")
        
        # Phase 6: Compile statistics
        print("\n[PHASE 6] FASE 6: Cálculo de Estadísticas...")
        stats = self._compute_statistics(train_data, val_data, test_data)
        
        # Create final data structure
        inductive_data = INDUSTRYInductiveData(
            # Training
            train_edge_index=train_data['edge_index'],
            train_edge_weights=train_data['edge_weights'],  # ✅ NEW
            train_loc_ids=train_data['loc_ids'],
            train_patrimonio_ids=train_data['patrimonio_ids'],
            train_patrimonio_tipo_ids=train_data['patrimonio_tipo_ids'],
            train_num_locs=train_data['num_locs'],
            train_num_patrimonios=train_data['num_patrimonios'],
            
            # Validation
            val_edge_index=val_data['edge_index'],
            val_edge_weights=val_data['edge_weights'],
            val_loc_ids=val_data['loc_ids'],
            val_patrimonio_ids=val_data['patrimonio_ids'],
            val_patrimonio_tipo_ids=val_data['patrimonio_tipo_ids'],
            val_num_locs=val_data['num_locs'],
            val_num_patrimonios=val_data['num_patrimonios'],
            
            # Test
            test_edge_index=test_data['edge_index'],
            test_edge_weights=test_data['edge_weights'],
            test_loc_ids=test_data['loc_ids'],
            test_patrimonio_ids=test_data['patrimonio_ids'],
            test_patrimonio_tipo_ids=test_data['patrimonio_tipo_ids'],
            test_num_locs=test_data['num_locs'],
            test_num_patrimonios=test_data['num_patrimonios'],
            
            vocabulary=vocabulary,
            stats=stats
        )
        
        # Print final summary
        self._print_summary(inductive_data)
        
        return inductive_data
    
    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load Excel and prepare data."""
        df = pd.read_excel(self.excel_path)
        
        # Parse dates
        df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')
        
        # Remove rows with missing critical columns
        df = df.dropna(subset=[self.date_column, self.loc_column, self.patrimonio_column, self.tipo_column])
        
        # Sort by date (oldest first)
        df = df.sort_values(self.date_column).reset_index(drop=True)
        
        print(f"  [OK] Loaded {len(df)} interactions")
        print(f"  [OK] Date range: {df[self.date_column].min()} to {df[self.date_column].max()}")
        
        return df
    
    def _chronological_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data chronologically."""
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df
    
    def _build_vocabularies(self, train_df: pd.DataFrame) -> INDUSTRYVocabulary:
        """Build vocabularies from training data only."""
        # Get unique values from training data
        unique_locs = sorted(train_df[self.loc_column].unique())
        unique_patrimonios = sorted(train_df[self.patrimonio_column].unique())
        unique_tipos = sorted(train_df[self.tipo_column].unique())
        
        # Create mappings (reserve index 0 for <UNK>)
        loc_to_idx = {'<UNK_LOC>': 0}
        loc_to_idx.update({loc: i+1 for i, loc in enumerate(unique_locs)})
        
        patrimonio_to_idx = {'<UNK_PATRIMONIO>': 0}
        patrimonio_to_idx.update({pat: i+1 for i, pat in enumerate(unique_patrimonios)})
        
        tipo_to_idx = {'<UNK_TIPO>': 0}
        tipo_to_idx.update({tipo: i+1 for i, tipo in enumerate(unique_tipos)})
        
        # Reverse mappings
        idx_to_loc = {v: k for k, v in loc_to_idx.items()}
        idx_to_patrimonio = {v: k for k, v in patrimonio_to_idx.items()}
        idx_to_tipo = {v: k for k, v in tipo_to_idx.items()}
        
        return INDUSTRYVocabulary(
            loc_to_idx=loc_to_idx,
            patrimonio_to_idx=patrimonio_to_idx,
            tipo_to_idx=tipo_to_idx,
            idx_to_loc=idx_to_loc,
            idx_to_patrimonio=idx_to_patrimonio,
            idx_to_tipo=idx_to_tipo,
            unk_loc_idx=0,
            unk_patrimonio_idx=0,
            unk_tipo_idx=0,
            num_locs=len(loc_to_idx),
            num_patrimonios=len(patrimonio_to_idx),
            num_tipos=len(tipo_to_idx)
        )
    
    def _get_observed_nodes(
        self, train_df: pd.DataFrame
    ) -> Tuple[set, set]:
        """Get sets of observed nodes from training data."""
        observed_locs = set(train_df[self.loc_column].unique())
        observed_patrimonios = set(train_df[self.patrimonio_column].unique())
        return observed_locs, observed_patrimonios
    
    def _process_training_data(
        self, train_df: pd.DataFrame, vocabulary: INDUSTRYVocabulary
    ) -> Dict:
        """Process training data (all nodes are observed)."""
        # ✅ COMPUTE EDGE WEIGHTS: Count edge frequencies BEFORE deduplication
        edge_counts = train_df.groupby([self.loc_column, self.patrimonio_column]).size().to_dict()
        
        # ✅ APLICAR drop_duplicates: Solo queremos aristas únicas por (LOC, PATRIMONIO)
        # Elimina transacciones repetidas del mismo par
        train_edges_df = train_df[[self.loc_column, self.patrimonio_column, self.tipo_column]].drop_duplicates(
            subset=[self.loc_column, self.patrimonio_column]
        )
        
        # Map to indices
        loc_indices = train_edges_df[self.loc_column].map(vocabulary.loc_to_idx).values
        patrimonio_indices = train_edges_df[self.patrimonio_column].map(vocabulary.patrimonio_to_idx).values
        
        # Get unique node IDs
        unique_locs = np.unique(loc_indices)
        unique_patrimonios = np.unique(patrimonio_indices)
        
        # Create edge index (bipartite: loc -> patrimonio)
        # Remap to contiguous indices
        loc_remap = {old: new for new, old in enumerate(unique_locs)}
        patrimonio_remap = {old: new for new, old in enumerate(unique_patrimonios)}
        
        edges_u = np.array([loc_remap[loc] for loc in loc_indices])
        edges_v = np.array([patrimonio_remap[pat] + len(unique_locs) for pat in patrimonio_indices])
        
        edge_index = torch.LongTensor(np.vstack([edges_u, edges_v]))
        
        # Node ID tensors (for embedding lookup)
        loc_ids = torch.LongTensor(unique_locs)
        patrimonio_ids = torch.LongTensor(unique_patrimonios)
        
        # ✅ Create TIPO attribute tensor (maps each patrimonio to its tipo)
        # Build mapping from patrimonio -> tipo
        patrimonio_to_tipo = {}
        for _, row in train_edges_df.iterrows():
            pat = row[self.patrimonio_column]
            tipo = row[self.tipo_column]
            pat_idx = vocabulary.patrimonio_to_idx[pat]
            tipo_idx = vocabulary.tipo_to_idx[tipo]
            # Each patrimonio should have only one tipo
            if pat_idx not in patrimonio_to_tipo:
                patrimonio_to_tipo[pat_idx] = tipo_idx
        
        # Create tensor with tipo for each unique patrimonio in this split
        patrimonio_tipo_ids = torch.LongTensor([
            patrimonio_to_tipo[pat_idx.item()] for pat_idx in patrimonio_ids
        ])
        
        # ✅ Create edge_weights tensor (frequency of each edge in TRAIN)
        edge_weights = torch.FloatTensor([
            edge_counts[(train_edges_df.iloc[i][self.loc_column], 
                        train_edges_df.iloc[i][self.patrimonio_column])]
            for i in range(len(train_edges_df))
        ])
        
        print(f"  [OK] {len(unique_locs)} LOCALIZAÇÕEs únicas")
        print(f"  [OK] {len(unique_patrimonios)} OBSERVAÇÕES PATRIMÔNIO únicas")
        print(f"  [OK] {len(patrimonio_to_tipo)} PATRIMONIO->TIPO mappings")
        print(f"  [OK] {edge_index.size(1)} enlaces")
        print(f"  [OK] Edge weights: min={edge_weights.min():.1f}, max={edge_weights.max():.1f}, mean={edge_weights.mean():.2f}")
        
        return {
            'edge_index': edge_index,
            'edge_weights': edge_weights,  # ✅ NEW: edge weights from TRAIN
            'loc_ids': loc_ids,
            'patrimonio_ids': patrimonio_ids,
            'patrimonio_tipo_ids': patrimonio_tipo_ids,
            'num_locs': len(unique_locs),
            'num_patrimonios': len(unique_patrimonios),
            'new_locs': 0,  # No new nodes in training
            'new_patrimonios': 0
        }
    
    def _process_split_data(
        self,
        df: pd.DataFrame,
        vocabulary: INDUSTRYVocabulary,
        train_data: Dict,
        split_name: str
    ) -> Dict:
        """Process validation or test data (may contain new nodes mapped to <UNK>)."""
        # ✅ APLICAR drop_duplicates: Solo queremos aristas únicas por (LOC, PATRIMONIO)
        edges_df = df[[self.loc_column, self.patrimonio_column, self.tipo_column]].drop_duplicates(
            subset=[self.loc_column, self.patrimonio_column]
        )
        
        # Map to indices (use <UNK> for new nodes)
        loc_indices = []
        patrimonio_indices = []
        tipo_indices = []
        
        for _, row in edges_df.iterrows():
            loc = row[self.loc_column]
            patrimonio = row[self.patrimonio_column]
            tipo = row[self.tipo_column]
            
            # Use <UNK> if not in vocabulary
            loc_idx = vocabulary.loc_to_idx.get(loc, vocabulary.unk_loc_idx)
            patrimonio_idx = vocabulary.patrimonio_to_idx.get(patrimonio, vocabulary.unk_patrimonio_idx)
            tipo_idx = vocabulary.tipo_to_idx.get(tipo, vocabulary.unk_tipo_idx)
            
            loc_indices.append(loc_idx)
            patrimonio_indices.append(patrimonio_idx)
            tipo_indices.append(tipo_idx)
        
        loc_indices = np.array(loc_indices)
        patrimonio_indices = np.array(patrimonio_indices)
        tipo_indices = np.array(tipo_indices)
        
        # Get unique node IDs
        unique_locs = np.unique(loc_indices)
        unique_patrimonios = np.unique(patrimonio_indices)
        
        # Create edge index
        loc_remap = {old: new for new, old in enumerate(unique_locs)}
        patrimonio_remap = {old: new for new, old in enumerate(unique_patrimonios)}
        
        edges_u = np.array([loc_remap[loc] for loc in loc_indices])
        edges_v = np.array([patrimonio_remap[pat] + len(unique_locs) for pat in patrimonio_indices])
        
        edge_index = torch.LongTensor(np.vstack([edges_u, edges_v]))
        
        # Node ID tensors
        loc_ids = torch.LongTensor(unique_locs)
        patrimonio_ids = torch.LongTensor(unique_patrimonios)
        
        # ✅ Create TIPO attribute tensor for patrimonios in this split
        patrimonio_to_tipo = {}
        for i, pat_idx in enumerate(patrimonio_indices):
            tipo_idx = tipo_indices[i]
            if pat_idx not in patrimonio_to_tipo:
                patrimonio_to_tipo[pat_idx] = tipo_idx
        
        # Create tensor with tipo for each unique patrimonio
        patrimonio_tipo_ids = torch.LongTensor([
            patrimonio_to_tipo[pat_idx.item()] for pat_idx in patrimonio_ids
        ])
        
        # Count new nodes (nodes not in vocabulary, i.e., mapped to <UNK>)
        new_locs = int(torch.sum(loc_ids == vocabulary.unk_loc_idx).item())
        new_patrimonios = int(torch.sum(patrimonio_ids == vocabulary.unk_patrimonio_idx).item())
        
        # Compute edge frequencies from this split's own data (not from TRAIN)
        split_edge_freq = df.groupby([self.loc_column, self.patrimonio_column]).size().to_dict()
        
        # Map edge weights for current split
        edges_locs = edges_df[self.loc_column].tolist()
        edges_pats = edges_df[self.patrimonio_column].tolist()
        edge_weights_list = [
            float(split_edge_freq.get((ls, ps), 1.0))
            for ls, ps in zip(edges_locs, edges_pats)
        ]
        edge_weights = torch.FloatTensor(edge_weights_list)
        
        # Count new edges: pairs not seen in TRAIN
        train_ei = train_data['edge_index'].numpy()
        train_li = train_data['loc_ids'].numpy()
        train_pi = train_data['patrimonio_ids'].numpy()
        train_pair_set = set()
        for j in range(train_ei.shape[1]):
            loc_v = int(train_li[train_ei[0, j]])
            pat_v = int(train_pi[train_ei[1, j] - len(train_li)])
            train_pair_set.add((
                vocabulary.idx_to_loc.get(loc_v, ''),
                vocabulary.idx_to_patrimonio.get(pat_v, '')
            ))
        num_new_edges = sum(1 for ls, ps in zip(edges_locs, edges_pats) if (ls, ps) not in train_pair_set)
        
        print(f"  [OK] {len(unique_locs)} LOCALIZAÇÕEs únicas ({new_locs} nuevas -> <UNK>)")
        print(f"  [OK] {len(unique_patrimonios)} OBSERVAÇÕES PATRIMÔNIO únicas ({new_patrimonios} nuevas -> <UNK>)")
        print(f"  [OK] {edge_index.size(1)} enlaces ({num_new_edges} nuevos pares no vistos en TRAIN)")
        print(f"  [OK] Edge weights: min={edge_weights.min():.1f}, max={edge_weights.max():.1f}, mean={edge_weights.mean():.2f}")
        
        return {
            'edge_index': edge_index,
            'edge_weights': edge_weights,  # ✅ NEW: edge weights from train or 1.0
            'loc_ids': loc_ids,
            'patrimonio_ids': patrimonio_ids,
            'patrimonio_tipo_ids': patrimonio_tipo_ids,
            'num_locs': len(unique_locs),
            'num_patrimonios': len(unique_patrimonios),
            'new_locs': new_locs,
            'new_patrimonios': new_patrimonios
        }
    
    def _compute_statistics(
        self,
        train_data: Dict,
        val_data: Dict,
        test_data: Dict
    ) -> Dict[str, int]:
        """Compute comprehensive statistics."""
        return {
            'train_locs': train_data['num_locs'],
            'train_patrimonios': train_data['num_patrimonios'],
            'train_edges': train_data['edge_index'].size(1),
            
            'val_locs': val_data['num_locs'],
            'val_patrimonios': val_data['num_patrimonios'],
            'val_edges': val_data['edge_index'].size(1),
            'val_new_locs': val_data['new_locs'],
            'val_new_patrimonios': val_data['new_patrimonios'],
            
            'test_locs': test_data['num_locs'],
            'test_patrimonios': test_data['num_patrimonios'],
            'test_edges': test_data['edge_index'].size(1),
            'test_new_locs': test_data['new_locs'],
            'test_new_patrimonios': test_data['new_patrimonios'],
        }
    
    def _print_summary(self, data: INDUSTRYInductiveData):
        """Print final summary."""
        print("\n" + "=" * 80)
        print("[SUCCESS] CARGA DE DATOS COMPLETADA - VERSIÓN SIMPLIFICADA")
        print("=" * 80)
        print(f"\n[VOCAB] VOCABULARIO (construido solo desde PASADO):")
        print(f"  - LOCALIZAÇÕEs: {data.vocabulary.num_locs} (incluyendo <UNK>)")
        print(f"  - PATRIMÔNIOS:  {data.vocabulary.num_patrimonios} (incluyendo <UNK>)")
        print(f"  - TIPO DOs (attr): {data.vocabulary.num_tipos} (incluyendo <UNK>)")
        
        print(f"\n[PAST] PASADO - TRAINING SET (80%):")
        print(f"  - Nodos: {data.train_num_locs} LOCs + {data.train_num_patrimonios} PATRIMÔNIOs")
        print(f"  - Enlaces: {data.train_edge_index.size(1)}")
        print(f"  - Uso: Entrenar Encoder (auto-supervisado)")
        
        print(f"\n[PRESENT] PRESENTE - VALIDATION SET (10%):")
        print(f"  - Nodos: {data.val_num_locs} LOCs + {data.val_num_patrimonios} PATRIMÔNIOs")
        print(f"  - Nodos nuevos: {data.stats['val_new_locs']} LOCs + {data.stats['val_new_patrimonios']} PATRIMÔNIOs -> <UNK>")
        print(f"  - Enlaces: {data.val_edge_index.size(1)}")
        print(f"  - Uso: Entrenar Decoder (supervisado) con early stopping")
        
        print(f"\n[FUTURE] FUTURO - TEST SET (10%):")
        print(f"  - Nodos: {data.test_num_locs} LOCs + {data.test_num_patrimonios} PATRIMÔNIOs")
        print(f"  - Nodos nuevos: {data.stats['test_new_locs']} LOCs + {data.stats['test_new_patrimonios']} PATRIMÔNIOs -> <UNK>")
        print(f"  - Enlaces: {data.test_edge_index.size(1)}")
        print(f"  - Uso: Evaluación final global (predicción temporal)")
        
        print("\n" + "=" * 80)
