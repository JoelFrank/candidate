"""
Configuration utilities for experiments.
"""

import yaml
import json
from typing import Dict, Any, Optional
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        if save_path.endswith('.yaml') or save_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False)
        elif save_path.endswith('.json'):
            json.dump(config, f, indent=2)
        else:
            raise ValueError("Save path must be .yaml, .yml, or .json")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for experiments.
    
    Returns:
        Default configuration dictionary
    """
    config = {
        'model': {
            'name': 'TBGRL',
            'hidden_dim': 256,
            'output_dim': 128,
            'num_layers': 2,
            'dropout': 0.1,
            'tau': 0.995,
            'lambda_param': 0.5,
            'conv_type': 'gcn'
        },
        
        'data': {
            'dataset_name': 'synthetic',
            'evaluation_mode': 'inductive',
            'normalize_features': True,
            'split_params': {
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'test_ratio': 0.2,
                'observed_ratio': 0.7,
                'seed': 42
            }
        },
        
        'training': {
            'epochs': 1000,
            'lr': 0.001,
            'weight_decay': 5e-4,
            'patience': 100,
            'min_epochs': 50,
            'eval_freq': 10
        },
        
        'evaluation': {
            'k_values': [10, 50, 100],
            'scoring_function': 'dot_product',
            'metrics': ['hits@50', 'auc_roc', 'average_precision', 'mrr']
        },
        
        'logging': {
            'log_level': 'INFO',
            'save_embeddings': False,
            'save_model': True,
            'results_dir': 'results',
            'wandb_project': 'bipartite-ssl'
        }
    }
    
    return config


def update_config(base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        base_config: Base configuration
        updates: Updates to apply
        
    Returns:
        Updated configuration
    """
    def deep_update(d: Dict, u: Dict) -> Dict:
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    import copy
    updated_config = copy.deepcopy(base_config)
    return deep_update(updated_config, updates)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration for required fields.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = [
        'model.name',
        'model.hidden_dim',
        'model.output_dim',
        'data.evaluation_mode',
        'training.epochs',
        'training.lr'
    ]
    
    def get_nested_value(d: Dict, key: str):
        keys = key.split('.')
        value = d
        for k in keys:
            if k not in value:
                return None
            value = value[k]
        return value
    
    for field in required_fields:
        value = get_nested_value(config, field)
        if value is None:
            raise ValueError(f"Required field missing: {field}")
    
    # Validate model name
    valid_models = ['TBGRL', 'BGRL', 'GBT', 'CCASSG']
    if config['model']['name'] not in valid_models:
        raise ValueError(f"Invalid model name. Must be one of: {valid_models}")
    
    # Validate evaluation mode
    valid_modes = ['inductive', 'transductive']
    if config['data']['evaluation_mode'] not in valid_modes:
        raise ValueError(f"Invalid evaluation mode. Must be one of: {valid_modes}")
    
    return True


def create_experiment_config(
    model_name: str,
    dataset_name: str = 'synthetic',
    evaluation_mode: str = 'inductive',
    **kwargs
) -> Dict[str, Any]:
    """
    Create experiment configuration with specific parameters.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        evaluation_mode: Evaluation mode
        **kwargs: Additional parameters to override
        
    Returns:
        Experiment configuration
    """
    config = get_default_config()
    
    # Update with specified parameters
    config['model']['name'] = model_name
    config['data']['dataset_name'] = dataset_name
    config['data']['evaluation_mode'] = evaluation_mode
    
    # Apply additional overrides
    config = update_config(config, kwargs)
    
    # Validate configuration
    validate_config(config)
    
    return config
