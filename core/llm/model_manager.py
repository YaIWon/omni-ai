"""
MANAGES MULTIPLE MODELS AND SWITCHING
"""
import json
from typing import Dict, List
import numpy as np

class ModelManager:
    def __init__(self):
        self.active_model = None
        self.available_models = {
            "phi-2": "microsoft/phi-2",
            "phi-1.5": "microsoft/phi-1_5", 
            "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "qwen-1.8b": "Qwen/Qwen-1_8B-Chat",
            "stablelm-3b": "stabilityai/stablelm-3b-4e1t"
        }
        self.model_performance = {}
        self.load_model_configs()
    
    def load_model_configs(self):
        """Load model configurations"""
        config_path = Path("config/models.json")
        if config_path.exists():
            with open(config_path) as f:
                self.available_models.update(json.load(f))
    
    def select_best_model(self, task_type: str) -> str:
        """Select model based on task and performance history"""
        # Simple heuristic - can evolve
        if "code" in task_type.lower():
            return "phi-2"  # Good for code
        elif "reasoning" in task_type.lower():
            return "phi-1.5"
        else:
            return "tinyllama"  # General purpose
    
    def update_performance(self, model_name: str, success: bool, latency: float):
        """Track model performance for evolution"""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {
                "total_uses": 0,
                "successes": 0,
                "avg_latency": 0,
                "last_used": None
            }
        
        stats = self.model_performance[model_name]
        stats["total_uses"] += 1
        if success:
            stats["successes"] += 1
        stats["avg_latency"] = (stats["avg_latency"] * (stats["total_uses"] - 1) + latency) / stats["total_uses"]
        
        # Save to disk
        self._save_performance_stats()
    
    def evolve_model_selection(self):
        """Improve model selection based on performance"""
        # Could implement reinforcement learning here
        pass
