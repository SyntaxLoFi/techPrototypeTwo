"""
Strategy Loader - Dynamically loads all available strategies

This module discovers and loads all strategy implementations
from the strategies directory structure.
"""

import os
import importlib
import inspect
import logging
import config_manager
from typing import Dict, List, Type
try:
    from .base_strategy import BaseStrategy
except (ImportError, ValueError):
    from base_strategy import BaseStrategy


class StrategyLoader:
    """
    Dynamically loads all strategy implementations.
    """
    
    def __init__(self):
        self.strategies = {}
        self._load_all_strategies()
    
    def _load_all_strategies(self):
        """
        Discover and load all strategy classes.
        """
        # Get the strategies directory path
        strategies_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define subdirectories to search
        subdirs = ['options', 'perpetuals', 'hybrid']
        
        for subdir in subdirs:
            subdir_path = os.path.join(strategies_dir, subdir)
            if not os.path.exists(subdir_path):
                continue
            
            # Find all Python files in the subdirectory
            for filename in os.listdir(subdir_path):
                if filename.endswith('_strategy.py') and not filename.startswith('base_'):
                    module_name = filename[:-3]  # Remove .py
                    
                    try:
                        # Import the module
                        module_path = f'strategies.{subdir}.{module_name}'
                        module = importlib.import_module(module_path)
                        
                        # Find strategy classes in the module
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, BaseStrategy) and 
                                obj != BaseStrategy and
                                not name.startswith('Base')):
                                
                                # Create a unique key for the strategy
                                strategy_key = f"{subdir}.{name}"
                                self.strategies[strategy_key] = obj
                                
                    except Exception as e:
                        print(f"Error loading strategy from {module_name}: {str(e)}")
    
    def get_all_strategies(self) -> Dict[str, Type[BaseStrategy]]:
        """
        Get all loaded strategy classes.
        """
        return self.strategies
    
    def get_strategies_by_type(self, strategy_type: str) -> List[Type[BaseStrategy]]:
        """
        Get strategies of a specific type (options, perpetuals, hybrid).
        """
        return [
            strategy_class for key, strategy_class in self.strategies.items()
            if key.startswith(f"{strategy_type}.")
        ]
    
    def instantiate_all_strategies(self, risk_free_rate: float = 0.05) -> List[BaseStrategy]:
        """
        Create instances of all strategies. Tries to satisfy common constructor
        signatures: (cfg/config, logger, risk_free_rate).
        """
        instances: List[BaseStrategy] = []
        cfg = config_manager.get_config()
        for strategy_class in self.strategies.values():
            try:
                sig = inspect.signature(strategy_class.__init__)
                params = sig.parameters
                kwargs = {}
                if 'risk_free_rate' in params:
                    kwargs['risk_free_rate'] = risk_free_rate
                if 'cfg' in params:
                    kwargs['cfg'] = cfg
                if 'config' in params:
                    kwargs['config'] = cfg
                if 'logger' in params:
                    kwargs['logger'] = logging.getLogger(f"strategy.{strategy_class.__name__}")
                instance = strategy_class(**kwargs) if kwargs else strategy_class()
                instances.append(instance)
            except Exception as e:
                print(f"Error instantiating {strategy_class.__name__}: {e}")
        return instances
    
    def get_strategy_info(self) -> List[Dict]:
        """
        Get information about all loaded strategies.
        """
        info: List[Dict] = []
        cfg = config_manager.get_config()
        for key, strategy_class in self.strategies.items():
            try:
                sig = inspect.signature(strategy_class.__init__)
                params = sig.parameters
                kwargs = {}
                if 'risk_free_rate' in params:
                    kwargs['risk_free_rate'] = 0.05
                if 'cfg' in params:
                    kwargs['cfg'] = cfg
                if 'config' in params:
                    kwargs['config'] = cfg
                if 'logger' in params:
                    kwargs['logger'] = logging.getLogger(f"strategy.{strategy_class.__name__}")
                instance = strategy_class(**kwargs) if kwargs else strategy_class()
                info.append({
                    'key': key,
                    'class_name': strategy_class.__name__,
                    'name': instance.get_strategy_name(),
                    'type': instance.get_strategy_type(),
                    'module': strategy_class.__module__,
                })
            except Exception:
                pass
        return info

def load_strategies() -> List[object]:
    """Legacy convenience loader.

    NOTE: BinaryReplicationStrategy and BreedenLitzenbergerStrategy are deprecated and removed.
    Prefer using StrategyLoader(...).discover() for dynamic discovery. This function now returns
    only the maintained options strategy.
    """
    from strategies.options.variance_swap_strategy import VarianceSwapStrategy
    cfg = config_manager.get_config()
    return [
        VarianceSwapStrategy(config=cfg),
    ]