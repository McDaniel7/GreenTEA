# GreenTEA/config/config.py

import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

@dataclass
class Config:
    # File paths
    dev_file: str
    test_file: str
    rootpath: str
    
    # Basic configuration
    task: str
    dataset: str
    testing: bool
    dev_sample_num: int
    test_sample_num: int
    
    # Output paths
    output_base_path: str
    output_path: str
    cache_path: str
    
    # Input files
    manual_prompt_file: str
    input_idx_list_file: str
    sbert_model_file: str
    manual_prompt_file_path: str
    input_idx_list_file_path: str
    sbert_model_file_path: str
    llm_prompt_file: Optional[str]
    llm_prompt_file_path: str
    
    # Model paths
    investigator_prompt_template_path: str
    investigator_system_prompt_path: str
    error_hunter_prompt_path: str
    crossover_prompt_path: str
    rephraser_template_path: str
    
    # Device configuration
    device: str
    
    # Evolution parameters
    pop_size: int
    budget: int
    batch_size: int
    seed: int
    parent_selection: str
    init_mode: str
    init_pop: str
    task_prompt: bool
    gradient_guided: bool
    
    # Example collection configuration
    exp_collect_mode: str
    exp_collect_n_max: int
    
    # Model names
    investigator_model_name: str
    error_hunter_model_name: str
    child_generator_model_name: str
    paraphraser_model_name: str
    
    # LLM configurations
    investigator_config: Dict[str, Any]
    error_hunter_config: Dict[str, Any]
    child_generator_config: Dict[str, Any]
    paraphraser_config: Dict[str, Any]
    
    # Feature configuration
    input_idx_list: List[str]
    
    # Saving configuration
    logger_name: str
    model_identifier: str
    save_file_name: str

    # Checkpoint configuration
    ckpt_pop: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a Config instance from a dictionary"""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """Create a Config instance from a JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config instance to dictionary"""
        return asdict(self)

    def save(self, json_path: str) -> None:
        """Save Config instance to a JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def __post_init__(self):
        """Validate configuration after initialization"""
        # Validate evolution parameters
        if self.pop_size <= 0:
            raise ValueError("Population size must be positive")
        if self.budget <= 0:
            raise ValueError("Budget must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
            
        # Validate parent selection method
        if self.parent_selection not in ["wheel", "random"]:
            raise ValueError("Invalid parent selection method")
            
        # Validate initialization mode
        if self.init_mode not in ["all", "ckpt"]:
            raise ValueError("Invalid initialization mode")
            
        # Validate initial population selection
        if self.init_pop not in ["topk", "bottomk", "randomk", "para_topk", "para_bottomk", "para_randomk"]:
            raise ValueError("Invalid initial population selection method")
            
        # Validate example collection mode
        if self.gradient_guided and self.exp_collect_mode not in ["random", "score_d", "topic"]:
            raise ValueError("Invalid example collection mode")
            
        # Validate device
        if self.device not in ["cpu", "cuda"]:
            raise ValueError("Invalid device specified")
            
    # support square bracket notation for access the attributes
    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to configuration attributes"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Configuration has no attribute '{key}'")
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Enable dictionary-style setting of configuration attributes"""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Configuration has no attribute '{key}'")
            
    def keys(self) -> List[str]:
        """Return list of configuration keys"""
        return list(self.to_dict().keys())
    
    def items(self) -> List[tuple]:
        """Return list of configuration items"""
        return list(self.to_dict().items())
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style get method with default value"""
        try:
            return self[key]
        except KeyError:
            return default
