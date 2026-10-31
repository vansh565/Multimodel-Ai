import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import random

class RLOptimizer:
    """Reinforcement Learning optimizer for retrieval strategies"""
    
    def __init__(self, config):
        self.config = config
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.strategies = self._initialize_strategies()
        self.learning_rate = config.RL_LEARNING_RATE
        self.gamma = config.RL_GAMMA
        self.epsilon = config.RL_EPSILON
        
    def _initialize_strategies(self) -> List[Dict]:
        """Initialize possible retrieval strategies"""
        return [
            {'name': 'balanced', 'local_mult': 1.0, 'global_mult': 1.0, 'local_k': 3, 'global_k': 2},
            {'name': 'local_focused', 'local_mult': 1.5, 'global_mult': 0.5, 'local_k': 5, 'global_k': 1},
            {'name': 'global_focused', 'local_mult': 0.5, 'global_mult': 1.5, 'local_k': 2, 'global_k': 4},
            {'name': 'precise', 'local_mult': 1.2, 'global_mult': 0.8, 'local_k': 4, 'global_k': 1},
            {'name': 'broad', 'local_mult': 0.7, 'global_mult': 1.3, 'local_k': 2, 'global_k': 3}
        ]
    
    def get_strategy(self, session_id: str) -> Dict:
        """Get best strategy using epsilon-greedy"""
        if random.random() < self.epsilon:
            # Explore
            return random.choice(self.strategies)
        else:
            # Exploit - get best strategy for this session
            return self._get_best_strategy(session_id)
    
    def _get_best_strategy(self, session_id: str) -> Dict:
        """Get strategy with highest Q-value"""
        strategy_values = self.q_table[session_id]
        if not strategy_values:
            return self.strategies[0]
        
        best_strategy_name = max(strategy_values, key=strategy_values.get)
        for strategy in self.strategies:
            if strategy['name'] == best_strategy_name:
                return strategy
        return self.strategies[0]
    
    def update(self, session_id: str, features: Dict, reward: float):
        """Update Q-values based on reward"""
        current_strategy = self._get_best_strategy(session_id)
        current_q = self.q_table[session_id][current_strategy['name']]
        
        # Simple Q-learning update
        new_q = current_q + self.learning_rate * (reward - current_q)
        self.q_table[session_id][current_strategy['name']] = new_q
        
        # Adjust epsilon (exploration rate) for this session
        self._adjust_epsilon(session_id)
    
    def _adjust_epsilon(self, session_id: str):
        """Gradually reduce exploration rate"""
        interactions = len(self.q_table[session_id])
        if interactions > 50:
            self.epsilon = max(0.05, self.epsilon * 0.99)