# GreenTEA/data/data_processor.py

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from tqdm import tqdm

class DataPreparer(ABC):
    """Base class for data preparation"""
    
    def __init__(self, answer_is_label: Optional[bool] = None):
        """
        Initialize data preparer.
        
        Args:
        """
        self.answer_is_label = answer_is_label

    @abstractmethod
    def _extract_annotation(self, text: str) -> str:
        """
        Extract annotation from text.
        
        Args:
            text: Input text to extract annotation from
            
        Returns:
            Extracted annotation
        """
        pass

    @abstractmethod
    def _extract_label(self, text: str) -> int:
        """
        Extract label from text.
        
        Args:
            text: Input text to extract label from
            
        Returns:
            Extracted label (0 or 1)
        """
        pass

    def forward(self, df: pd.DataFrame,
                desc: str = "Processing data") -> Tuple[List[str], List[str], List[int]]:
        """
        Process data and return input list, annotation list, and label list.
        
        Args:
            df: DataFrame containing the data
            desc: Description for progress bar
            
        Returns:
            Tuple containing:
            - List of input questions
            - List of true annotations
            - List of true labels
        """
        input_list = df['question'].tolist()
        
        if self.answer_is_label():
            # Answer column contains labels
            true_label_list = df['answer'].tolist()
            true_annotation_list = [
                self._extract_annotation(q) 
                for q in tqdm(df['question'], desc=desc)
            ]
        else:
            # Answer column contains annotations
            true_annotation_list = df['answer'].tolist()
            true_label_list = [
                self._extract_label(a) 
                for a in tqdm(df['answer'], desc=desc)
            ]
            
        return input_list, true_annotation_list, true_label_list