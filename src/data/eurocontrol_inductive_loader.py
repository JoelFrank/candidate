"""
EuroControl Inductive-Temporal Data Loader

Dataset: EuroControl flight list (Airport-Airline bipartite graph)

Graph Definition:
  - Node type U: Airport (derived from BOTH adep and ades columns)
  - Node type V: Airline (icao_operator)
  - Edge U-V: Airport <> Airline connection (flight co-occurrence)
  - Edge weight: Number of flights between airport and airline (departures + arrivals)

Node Features:
  Airport (U nodes - 5 features):
    - airport_total_flights     : total flights involving this airport (dep + arr)
    - airport_num_departures    : flights where airport is adep
    - airport_num_arrivals      : flights where airport is ades
    - airport_unique_airlines   : number of unique icao_operators
    - airport_unique_aircraft   : number of unique icao24 values

  Airline (V nodes - 4 features):
    - airline_total_flights     : total number of flights
    - airline_unique_airports   : number of unique airports served (dep + arr)
    - airline_unique_aircraft   : number of unique icao24 values
    - airline_fleet_diversity   : number of unique typecodes

Inductive-Temporal Protocol:
  1. Chronological split by 'dof': 80% PAST / 10% PRESENT / 10% FUTURE
  2. Vocabulary (airport_to_idx, airline_to_idx) built ONLY from PAST
  3. Features computed with temporal restrictions:
     - PAST features use only PAST history
     - PRESENT features use only PAST history (no leakage)
     - FUTURE features use PAST + PRESENT history
     - Scaling: log1p compression THEN MinMaxScaler, fit ONLY on TRAIN
       Val/Test use .transform() with the TRAIN-fitted scaler (no leakage)
  4. New nodes in val/test mapped to UNK (zero features)
  5. Edge weights per split reflect ACTUAL flight counts in that time window
  6. Negative sampling excludes ALL real edges (train + val + test)

Columns used: dof, adep, ades, icao_operator, icao24, typecode
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


AIRPORT_FEAT_DIM = 5
AIRLINE_FEAT_DIM = 4


@dataclass
class EuroControlVocabulary:
    """Vocabularies for EuroControl nodes - built only from TRAIN."""
    airport_to_idx: Dict[str, int]   # airport code -> index (0 = UNK)
    airline_to_idx: Dict[str, int]   # icao_operator -> index (0 = UNK)
    idx_to_airport: Dict[int, str]
    idx_to_airline: Dict[int, str]
    unk_airport_idx: int             # Always 0
    unk_airline_idx: int             # Always 0
    num_airports: int                # Including <UNK>
    num_airlines: int                # Including <UNK>


@dataclass
class EuroControlInductiveData:
    """Complete inductive data structure for EuroControl."""

    # Training data (PAST - 80%)
    train_airport_ids: np.ndarray          # Airport codes in train
    train_airline_ids: np.ndarray          # Airline codes in train
    train_num_airports: int
    train_num_airlines: int
    train_edge_index: torch.LongTensor     # [2, num_train_edges]
    train_edge_weights: torch.FloatTensor  # [num_train_edges] - flight co-occurrence counts
    train_airport_features: torch.FloatTensor  # [num_airports, 5]
    train_airline_features: torch.FloatTensor  # [num_airlines, 4]

    # Validation data (PRESENT - 10%)
    val_airport_ids: np.ndarray
    val_airline_ids: np.ndarray
    val_num_airports: int
    val_num_airlines: int
    val_edge_index: torch.LongTensor
    val_edge_weights: torch.FloatTensor    # edge frequencies from VAL split period
    val_airport_features: torch.FloatTensor    # computed from PAST only
    val_airline_features: torch.FloatTensor    # computed from PAST only

    # Test data (FUTURE - 10%)
    test_airport_ids: np.ndarray
    test_airline_ids: np.ndarray
    test_num_airports: int
    test_num_airlines: int
    test_edge_index: torch.LongTensor
    test_edge_weights: torch.FloatTensor   # edge frequencies from TEST split period
    test_airport_features: torch.FloatTensor   # computed from PAST + PRESENT
    test_airline_features: torch.FloatTensor   # computed from PAST + PRESENT

    # Vocabulary (from TRAIN only)
    vocabulary: EuroControlVocabulary

    # Feature dimensions
    airport_feat_dim: int   # 5
    airline_feat_dim: int   # 4

    # Statistics
    stats: Dict[str, int]


class EuroControlInductiveLoader:
    """
    Loads EuroControl data following the INDUCTIVE TEMPORAL protocol.

    Key principles:
    1. Chronological splitting (80% train / 10% val / 10% test) by 'dof'
    2. Vocabulary built ONLY from training data (PAST)
    3. Features computed with temporal restrictions (no future leakage)
    4. New nodes in val/test: zero features (inductive setting)
    5. Edge weights = flight co-occurrence frequencies (for WBT-BGRL variants)

    Timeline:
    - TRAIN (80%):   Past data  -> Train encoder self-supervised
    - VAL   (10%):   Present data -> Train decoder supervised with early stopping
    - TEST  (10%):   Future data  -> Final global evaluation

    Graph construction:
    - Each flight row contributes TWO edges:
        (adep, icao_operator) and (ades, icao_operator)
    - Duplicate (airport, airline) pairs are collapsed into one edge
    - Edge weight = total number of flights for that (airport, airline) pair
    """

    def __init__(
        self,
        csv_path: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        date_column: str = 'dof',
        adep_column: str = 'adep',
        ades_column: str = 'ades',
        airline_column: str = 'icao_operator',
        aircraft_column: str = 'icao24',
        typecode_column: str = 'typecode',
        seed: int = 42,
        verbose: bool = True
    ):
        self.csv_path = csv_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.date_column = date_column
        self.adep_column = adep_column
        self.ades_column = ades_column
        self.airline_column = airline_column
        self.aircraft_column = aircraft_column
        self.typecode_column = typecode_column
        self.seed = seed
        self.verbose = verbose

        np.random.seed(seed)
        torch.manual_seed(seed)

    def load(self) -> EuroControlInductiveData:
        """Load and process EuroControl data following inductive temporal protocol."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("EUROCONTROL INDUCTIVE TEMPORAL LOADER")
            print("=" * 80)

        # Phase 1: Load and chronologically split
        if self.verbose:
            print("\n[PHASE 1] Carga y Division Cronologica...")
        df = self._load_and_prepare_data()
        train_df, val_df, test_df = self._chronological_split(df)

        if self.verbose:
            print(f"  [OK] PASADO  (Train): {len(train_df):,} vuelos ({self.train_ratio:.0%})")
            print(f"  [OK] PRESENTE (Val):  {len(val_df):,} vuelos ({self.val_ratio:.0%})")
            print(f"  [OK] FUTURO  (Test):  {len(test_df):,} vuelos ({self.test_ratio:.0%})")

        # Phase 2: Build vocabulary from TRAIN only
        if self.verbose:
            print("\n[PHASE 2] Construccion de Vocabularios (solo PASADO)...")
        vocabulary = self._build_vocabulary(train_df)

        if self.verbose:
            print(f"  [OK] Aeropuertos: {vocabulary.num_airports - 1} conocidos + 1 <UNK>")
            print(f"  [OK] Aerolineas:  {vocabulary.num_airlines - 1} conocidas + 1 <UNK>")

        # Phase 3: Process training data
        if self.verbose:
            print("\n[PHASE 3] Procesamiento de datos de ENTRENAMIENTO (PASADO)...")
        train_data = self._process_training_data(train_df, vocabulary)

        # Phase 4: Process validation data (features from PAST only)
        if self.verbose:
            print("\n[PHASE 4] Procesamiento de datos de VALIDACION (PRESENTE)...")
        val_data = self._process_split_data(
            split_df=val_df,
            history_df=train_df,     # <-- solo historia PAST (sin leakage)
            train_data=train_data,
            vocabulary=vocabulary,
            split_name="Validacion"
        )

        # Phase 5: Process test data (features from PAST + PRESENT)
        if self.verbose:
            print("\n[PHASE 5] Procesamiento de datos de PRUEBA (FUTURO)...")
        history_tv = pd.concat([train_df, val_df], ignore_index=True)
        test_data = self._process_split_data(
            split_df=test_df,
            history_df=history_tv,   # <-- historia PAST + PRESENT
            train_data=train_data,
            vocabulary=vocabulary,
            split_name="Prueba"
        )

        # Phase 6: Statistics
        stats = self._compute_statistics(train_data, val_data, test_data)

        inductive_data = EuroControlInductiveData(
            # Train
            train_airport_ids=train_data['airport_ids'],
            train_airline_ids=train_data['airline_ids'],
            train_num_airports=train_data['num_airports'],
            train_num_airlines=train_data['num_airlines'],
            train_edge_index=train_data['edge_index'],
            train_edge_weights=train_data['edge_weights'],
            train_airport_features=train_data['airport_features'],
            train_airline_features=train_data['airline_features'],
            # Val
            val_airport_ids=val_data['airport_ids'],
            val_airline_ids=val_data['airline_ids'],
            val_num_airports=val_data['num_airports'],
            val_num_airlines=val_data['num_airlines'],
            val_edge_index=val_data['edge_index'],
            val_edge_weights=val_data['edge_weights'],
            val_airport_features=val_data['airport_features'],
            val_airline_features=val_data['airline_features'],
            # Test
            test_airport_ids=test_data['airport_ids'],
            test_airline_ids=test_data['airline_ids'],
            test_num_airports=test_data['num_airports'],
            test_num_airlines=test_data['num_airlines'],
            test_edge_index=test_data['edge_index'],
            test_edge_weights=test_data['edge_weights'],
            test_airport_features=test_data['airport_features'],
            test_airline_features=test_data['airline_features'],
            # Vocabulary & dims
            vocabulary=vocabulary,
            airport_feat_dim=AIRPORT_FEAT_DIM,
            airline_feat_dim=AIRLINE_FEAT_DIM,
            stats=stats,
        )

        if self.verbose:
            self._print_summary(inductive_data)

        return inductive_data

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load CSV, clean it, and sort chronologically."""
        df = pd.read_csv(self.csv_path)

        # Parse date
        df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')

        # Drop rows with missing critical columns
        required = [
            self.date_column, self.adep_column, self.ades_column,
            self.airline_column, self.aircraft_column, self.typecode_column
        ]
        df = df.dropna(subset=required).copy()

        # Normalize string columns
        for col in [self.adep_column, self.ades_column, self.airline_column,
                    self.aircraft_column, self.typecode_column]:
            df[col] = df[col].astype(str).str.strip().str.upper()

        # Remove rows where any critical string is empty after normalization
        for col in [self.adep_column, self.ades_column, self.airline_column]:
            df = df[df[col] != ''].copy()

        # Sort chronologically (CRITICAL for inductive-temporal protocol)
        df = df.sort_values(self.date_column).reset_index(drop=True)

        if self.verbose:
            print(f"  [OK] {len(df):,} vuelos cargados")
            print(f"  [OK] Rango de fechas: {df[self.date_column].min().date()} "
                  f"a {df[self.date_column].max().date()}")
            print(f"  [OK] Aeropuertos unicos (adep+ades): "
                  f"{pd.concat([df[self.adep_column], df[self.ades_column]]).nunique()}")
            print(f"  [OK] Aerolineas unicas: {df[self.airline_column].nunique()}")

        return df

    def _chronological_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split chronologically: 80% / 10% / 10%."""
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end   = int(n * (self.train_ratio + self.val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df   = df.iloc[train_end:val_end].copy()
        test_df  = df.iloc[val_end:].copy()

        return train_df, val_df, test_df

    def _build_vocabulary(self, train_df: pd.DataFrame) -> EuroControlVocabulary:
        """Build airport and airline vocabularies from TRAIN only. Index 0 = <UNK>."""
        # Airports come from BOTH adep and ades
        unique_airports = sorted(
            set(train_df[self.adep_column].unique()) |
            set(train_df[self.ades_column].unique())
        )
        unique_airlines = sorted(train_df[self.airline_column].unique())

        # Index 0 is reserved for <UNK>
        airport_to_idx = {'<UNK_AIRPORT>': 0}
        airport_to_idx.update({ap: i + 1 for i, ap in enumerate(unique_airports)})

        airline_to_idx = {'<UNK_AIRLINE>': 0}
        airline_to_idx.update({al: i + 1 for i, al in enumerate(unique_airlines)})

        idx_to_airport = {v: k for k, v in airport_to_idx.items()}
        idx_to_airline = {v: k for k, v in airline_to_idx.items()}

        return EuroControlVocabulary(
            airport_to_idx=airport_to_idx,
            airline_to_idx=airline_to_idx,
            idx_to_airport=idx_to_airport,
            idx_to_airline=idx_to_airline,
            unk_airport_idx=0,
            unk_airline_idx=0,
            num_airports=len(airport_to_idx),
            num_airlines=len(airline_to_idx),
        )

    # -------------------------------------------------------------------------
    # Feature computation  (temporal restriction enforced by caller)
    # -------------------------------------------------------------------------

    def _compute_airport_features_raw(
        self,
        airport_codes: np.ndarray,
        history_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Compute 5 airport RAW (unscaled) features from history_df.

        Features:
          0: airport_total_flights
          1: airport_num_departures
          2: airport_num_arrivals
          3: airport_unique_airlines
          4: airport_unique_aircraft
        """
        rows = []
        for ap in airport_codes:
            dep_mask = history_df[self.adep_column] == ap
            arr_mask = history_df[self.ades_column] == ap
            dep_df = history_df[dep_mask]
            arr_df = history_df[arr_mask]

            total_flights     = int(dep_mask.sum()) + int(arr_mask.sum())
            num_departures    = int(dep_mask.sum())
            num_arrivals      = int(arr_mask.sum())
            unique_airlines   = int(
                pd.concat([dep_df[self.airline_column],
                           arr_df[self.airline_column]]).nunique()
            )
            unique_aircraft   = int(
                pd.concat([dep_df[self.aircraft_column],
                           arr_df[self.aircraft_column]]).nunique()
            )

            rows.append([total_flights, num_departures, num_arrivals,
                         unique_airlines, unique_aircraft])

        return np.array(rows, dtype=np.float32)

    def _compute_airline_features_raw(
        self,
        airline_codes: np.ndarray,
        history_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Compute 4 airline RAW (unscaled) features from history_df.

        Features:
          0: airline_total_flights
          1: airline_unique_airports
          2: airline_unique_aircraft
          3: airline_fleet_diversity  (unique typecodes)
        """
        rows = []
        for al in airline_codes:
            mask = history_df[self.airline_column] == al
            df_al = history_df[mask]

            total_flights     = int(mask.sum())
            unique_airports   = int(
                pd.concat([df_al[self.adep_column],
                           df_al[self.ades_column]]).nunique()
            )
            unique_aircraft   = int(df_al[self.aircraft_column].nunique())
            fleet_diversity   = int(df_al[self.typecode_column].nunique())

            rows.append([total_flights, unique_airports, unique_aircraft, fleet_diversity])

        return np.array(rows, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Edge construction helpers
    # -------------------------------------------------------------------------

    def _build_edges_from_df(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract all (airport, airline) pairs from a flights dataframe.

        Each flight row contributes TWO pairs:
          - (adep, icao_operator)
          - (ades, icao_operator)

        Returns a DataFrame with columns ['airport', 'airline', 'count']
        where 'count' = number of flights for that pair (edge weight).
        """
        dep_pairs = df[[self.adep_column, self.airline_column]].copy()
        dep_pairs.columns = ['airport', 'airline']

        arr_pairs = df[[self.ades_column, self.airline_column]].copy()
        arr_pairs.columns = ['airport', 'airline']

        all_pairs = pd.concat([dep_pairs, arr_pairs], ignore_index=True)

        # Group by (airport, airline) -> count total flights
        edge_counts = (
            all_pairs
            .groupby(['airport', 'airline'])
            .size()
            .reset_index(name='count')
        )

        return edge_counts

    # -------------------------------------------------------------------------
    # Split processors
    # -------------------------------------------------------------------------

    def _process_training_data(
        self,
        train_df: pd.DataFrame,
        vocabulary: EuroControlVocabulary,
    ) -> Dict:
        """Process PAST (training) data. All nodes are in vocabulary."""

        # Build edge pairs with counts (BEFORE deduplication for weights)
        edge_counts_df = self._build_edges_from_df(train_df)

        # Map to vocabulary indices
        airport_vocab_indices = edge_counts_df['airport'].map(
            vocabulary.airport_to_idx
        ).fillna(vocabulary.unk_airport_idx).astype(int).values

        airline_vocab_indices = edge_counts_df['airline'].map(
            vocabulary.airline_to_idx
        ).fillna(vocabulary.unk_airline_idx).astype(int).values

        # Fallback to 1.0 if count is somehow 0 (spec requirement)
        raw_counts = edge_counts_df['count'].values.astype(np.float32)
        edge_weights_raw = np.where(raw_counts > 0, raw_counts, 1.0).astype(np.float32)

        # Unique nodes in training
        unique_airport_vocab = np.unique(airport_vocab_indices)
        unique_airline_vocab = np.unique(airline_vocab_indices)

        # Local remapping (contiguous indices within this split)
        airport_local = {v: i for i, v in enumerate(unique_airport_vocab)}
        airline_local  = {v: i for i, v in enumerate(unique_airline_vocab)}

        num_airports = len(unique_airport_vocab)

        edges_u = np.array([airport_local[v] for v in airport_vocab_indices])
        edges_v = np.array([airline_local[v] + num_airports for v in airline_vocab_indices])

        edge_index   = torch.LongTensor(np.vstack([edges_u, edges_v]))
        edge_weights = torch.FloatTensor(edge_weights_raw)

        # Node ID arrays (vocab indices, for reference)
        airport_ids = unique_airport_vocab.copy()
        airline_ids = unique_airline_vocab.copy()

        # Airport codes for feature computation
        airport_codes = np.array([vocabulary.idx_to_airport[idx] for idx in unique_airport_vocab])
        airline_codes = np.array([vocabulary.idx_to_airline[idx] for idx in unique_airline_vocab])

        # Compute RAW features from PAST history
        airport_raw = self._compute_airport_features_raw(airport_codes, train_df)
        airline_raw = self._compute_airline_features_raw(airline_codes, train_df)

        # Apply log1p compression (spec: log1p THEN MinMaxScaler)
        airport_log = np.log1p(airport_raw)
        airline_log  = np.log1p(airline_raw)

        # Fit scalers ONLY on TRAIN (will be reused for val/test via .transform())
        airport_scaler = MinMaxScaler()
        airline_scaler  = MinMaxScaler()

        airport_features = torch.from_numpy(
            airport_scaler.fit_transform(airport_log).astype(np.float32)
        )
        airline_features = torch.from_numpy(
            airline_scaler.fit_transform(airline_log).astype(np.float32)
        )

        if self.verbose:
            print(f"  [OK] {num_airports} aeropuertos unicos")
            print(f"  [OK] {len(unique_airline_vocab)} aerolineas unicas")
            print(f"  [OK] {edge_index.size(1)} enlaces (aristas unicas aeropuerto-aerolinea)")
            print(f"  [OK] Edge weights: min={edge_weights.min():.1f}, "
                  f"max={edge_weights.max():.1f}, mean={edge_weights.mean():.2f}")
            print(f"  [OK] Scalers ajustados en TRAIN (se aplicaran via .transform() en val/test)")

        return {
            'airport_ids':       airport_ids,
            'airline_ids':       airline_ids,
            'num_airports':      num_airports,
            'num_airlines':      len(unique_airline_vocab),
            'edge_index':        edge_index,
            'edge_weights':      edge_weights,
            'airport_features':  airport_features,
            'airline_features':  airline_features,
            # Fitted scalers — must be passed to val/test processing
            'airport_scaler':    airport_scaler,
            'airline_scaler':    airline_scaler,
            # Keep for weight mapping in val/test
            '_airport_vocab_indices': airport_vocab_indices,
            '_airline_vocab_indices': airline_vocab_indices,
            '_edge_weights_raw':      edge_weights_raw,
        }

    def _process_split_data(
        self,
        split_df: pd.DataFrame,
        history_df: pd.DataFrame,
        train_data: Dict,
        vocabulary: EuroControlVocabulary,
        split_name: str,
    ) -> Dict:
        """
        Process PRESENT (val) or FUTURE (test) data.

        - Nodes: union of all nodes seen in this split's edges
        - Features: computed using history_df (temporal restriction, NO leakage)
          * log1p applied, then transformed with TRAIN-fitted scalers (.transform() only)
        - Edges: only from split_df (unique airport-airline pairs in this period)
        - Edge weights: actual flight counts FROM THIS SPLIT PERIOD
        - New nodes (not in TRAIN vocab): get UNK index -> zero features after norm
        """

        # Build edge pairs for THIS split only
        split_edge_df = self._build_edges_from_df(split_df)

        # Map airports and airlines to vocabulary (UNK if not in train vocab)
        split_edge_df['airport_idx'] = split_edge_df['airport'].map(
            vocabulary.airport_to_idx
        ).fillna(vocabulary.unk_airport_idx).astype(int)

        split_edge_df['airline_idx'] = split_edge_df['airline'].map(
            vocabulary.airline_to_idx
        ).fillna(vocabulary.unk_airline_idx).astype(int)

        # All airport and airline vocab indices appearing in this split's edges
        airport_vocab_indices = split_edge_df['airport_idx'].values
        airline_vocab_indices = split_edge_df['airline_idx'].values

        # Unique nodes
        unique_airport_vocab = np.unique(airport_vocab_indices)
        unique_airline_vocab = np.unique(airline_vocab_indices)

        num_airports = len(unique_airport_vocab)

        # Local remapping
        airport_local = {v: i for i, v in enumerate(unique_airport_vocab)}
        airline_local  = {v: i for i, v in enumerate(unique_airline_vocab)}

        edges_u = np.array([airport_local[v] for v in airport_vocab_indices])
        edges_v = np.array([airline_local[v] + num_airports for v in airline_vocab_indices])

        edge_index = torch.LongTensor(np.vstack([edges_u, edges_v]))

        # Compute edge weights: use the actual flight counts FROM THIS SPLIT.
        # Each (airport, airline) pair already has its count from _build_edges_from_df.
        # Fallback to 1.0 if count is somehow 0 (spec requirement)
        raw_counts = split_edge_df['count'].values.astype(np.float32)
        edge_weights = torch.FloatTensor(np.where(raw_counts > 0, raw_counts, 1.0))

        num_new_airport = int(np.sum(unique_airport_vocab == vocabulary.unk_airport_idx))
        num_new_airline  = int(np.sum(unique_airline_vocab == vocabulary.unk_airline_idx))

        # Identify new edges: (airport, airline) pairs not seen in TRAIN
        train_weight_map = {}
        t_ap = train_data['_airport_vocab_indices']
        t_al = train_data['_airline_vocab_indices']
        t_w  = train_data['_edge_weights_raw']
        for i in range(len(t_w)):
            train_weight_map[(int(t_ap[i]), int(t_al[i]))] = True

        num_new_edges = int(sum(
            1 for i in range(len(split_edge_df))
            if (int(airport_vocab_indices[i]), int(airline_vocab_indices[i]))
               not in train_weight_map
        ))

        # Node ID arrays (vocab indices)
        airport_ids = unique_airport_vocab.copy()
        airline_ids  = unique_airline_vocab.copy()

        # Codes for feature computation
        # UNK nodes get an empty string — features will be zero (not in history_df)
        airport_codes = np.array([
            vocabulary.idx_to_airport.get(idx, '<UNK_AIRPORT>')
            for idx in unique_airport_vocab
        ])
        airline_codes = np.array([
            vocabulary.idx_to_airline.get(idx, '<UNK_AIRLINE>')
            for idx in unique_airline_vocab
        ])

        # Compute RAW features from history_df (temporal restriction enforced by caller)
        airport_raw = self._compute_airport_features_raw(airport_codes, history_df)
        airline_raw  = self._compute_airline_features_raw(airline_codes, history_df)

        # Apply log1p (same as train) then transform with TRAIN-fitted scalers
        # NO fit here — scalers were fit ONLY on train data
        airport_scaler = train_data['airport_scaler']
        airline_scaler  = train_data['airline_scaler']

        airport_features = torch.from_numpy(
            airport_scaler.transform(np.log1p(airport_raw)).astype(np.float32)
        )
        airline_features = torch.from_numpy(
            airline_scaler.transform(np.log1p(airline_raw)).astype(np.float32)
        )

        if self.verbose:
            print(f"  [OK] {num_airports} aeropuertos ({num_new_airport} nuevos -> <UNK>, features=0)")
            print(f"  [OK] {len(unique_airline_vocab)} aerolineas ({num_new_airline} nuevas -> <UNK>, features=0)")
            print(f"  [OK] {edge_index.size(1)} enlaces "
                  f"({num_new_edges} nuevos pares no vistos en TRAIN)")
            print(f"  [OK] Edge weights: min={edge_weights.min():.1f}, "
                  f"max={edge_weights.max():.1f}, mean={edge_weights.mean():.2f}")

        return {
            'airport_ids':      airport_ids,
            'airline_ids':       airline_ids,
            'num_airports':     num_airports,
            'num_airlines':      len(unique_airline_vocab),
            'edge_index':        edge_index,
            'edge_weights':      edge_weights,
            'airport_features':  airport_features,
            'airline_features':  airline_features,
            'new_airports':      num_new_airport,
            'new_airlines':      num_new_airline,
        }

    def _compute_statistics(
        self,
        train_data: Dict,
        val_data: Dict,
        test_data: Dict,
    ) -> Dict[str, int]:
        return {
            'train_airports':     train_data['num_airports'],
            'train_airlines':     train_data['num_airlines'],
            'train_edges':        train_data['edge_index'].size(1),

            'val_airports':       val_data['num_airports'],
            'val_airlines':       val_data['num_airlines'],
            'val_edges':          val_data['edge_index'].size(1),
            'val_new_airports':   val_data.get('new_airports', 0),
            'val_new_airlines':   val_data.get('new_airlines', 0),

            'test_airports':      test_data['num_airports'],
            'test_airlines':      test_data['num_airlines'],
            'test_edges':         test_data['edge_index'].size(1),
            'test_new_airports':  test_data.get('new_airports', 0),
            'test_new_airlines':  test_data.get('new_airlines', 0),
        }

    def _print_summary(self, data: EuroControlInductiveData):
        print("\n" + "=" * 80)
        print("[SUCCESS] CARGA DE DATOS COMPLETADA - EUROCONTROL")
        print("=" * 80)

        print(f"\n[VOCAB] VOCABULARIO (construido solo desde PASADO):")
        print(f"  - Aeropuertos: {data.vocabulary.num_airports} (incl. <UNK>)")
        print(f"  - Aerolineas:  {data.vocabulary.num_airlines} (incl. <UNK>)")

        print(f"\n[GRAPH] DIMENSIONES DE FEATURES:")
        print(f"  - Airport features: {data.airport_feat_dim} (total_flights, dep, arr, "
              f"unique_airlines, unique_aircraft)")
        print(f"  - Airline features: {data.airline_feat_dim} (total_flights, unique_airports, "
              f"unique_aircraft, fleet_diversity)")

        print(f"\n[PAST] PASADO - TRAINING SET (80%):")
        print(f"  - Nodos: {data.train_num_airports} aeropuertos + {data.train_num_airlines} aerolineas")
        print(f"  - Aristas: {data.train_edge_index.size(1)}")
        print(f"  - Uso: Entrenar Encoder (auto-supervisado)")

        print(f"\n[PRESENT] PRESENTE - VALIDATION SET (10%):")
        print(f"  - Nodos: {data.val_num_airports} aeropuertos + {data.val_num_airlines} aerolineas")
        print(f"  - Nodos nuevos: {data.stats['val_new_airports']} aeropuertos + "
              f"{data.stats['val_new_airlines']} aerolineas -> <UNK> (features=0)")
        print(f"  - Aristas: {data.val_edge_index.size(1)}")
        print(f"  - Uso: Entrenar Decoder (supervisado) con early stopping")

        print(f"\n[FUTURE] FUTURO - TEST SET (10%):")
        print(f"  - Nodos: {data.test_num_airports} aeropuertos + {data.test_num_airlines} aerolineas")
        print(f"  - Nodos nuevos: {data.stats['test_new_airports']} aeropuertos + "
              f"{data.stats['test_new_airlines']} aerolineas -> <UNK> (features=0)")
        print(f"  - Aristas: {data.test_edge_index.size(1)}")
        print(f"  - Uso: Evaluacion final global (prediccion temporal)")

        print("\n" + "=" * 80)
