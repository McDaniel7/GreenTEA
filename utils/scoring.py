# GreenTEA/models/evaluator.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
from sklearn import metrics

class AccEvaluator:
    """
    Accuracy evaluator for model outputs
    """
    def __init__(self):
        """
        Initialize accuracy evaluator.
        """

    def forward(self, model_decisions: List[str], true_labels: List[int]) -> List[float]:
        """
        Evaluate model outputs against true labels.
        
        Args:
            model_decisions: List of model predictions
            true_labels: List of true labels
            
        Returns:
            - List of evaluation metrics (Acc)
        """
        # Accuracy
        acc = np.mean([int(model_decision == true_label) for model_decision, true_label in zip(model_decisions, true_labels)])
        return [acc]