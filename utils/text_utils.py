# GreenTEA/utils/text_utils.py

import re
import random
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_tagged_text(text: str, start_tag: str, end_tag: str) -> str:
    """Parse text that is tagged with start and end tags."""
    try:
        start_index = text.find(start_tag)
        if start_index == -1:
            return ""
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            return ""
        start_index += len(start_tag)
        return text[start_index:end_index].strip()
    except Exception as e:
        return ""

def split_and_include_delimiters(text: str, delimiters: List[str]) -> tuple:
    """Split text by delimiters while keeping them"""
    pattern = '|'.join(map(re.escape, delimiters))
    splitted = re.split(pattern, text)
    delimits = re.findall(pattern, text)
    return splitted, delimits

def read_ckpt_files(file_path: str) -> List[str]:
    """Read checkpoint files"""
    with open(file_path, "r") as f:
        all_text = f.read()
        
    delimiters_list = ["manual\t", "evoluted\t", "llm\t", "best score: "]
    splitted, delimits = split_and_include_delimiters(all_text, delimiters_list)
    splitted = splitted[1:-1]
    delimits = delimits[:-1]
    
    assert len(splitted) == len(delimits)
    return [d + s for d, s in zip(delimits, splitted)]

def read_lines(file_: str, sample_indices: Optional[List[int]] = None) -> List[str]:
    """Read lines from a file"""
    with open(file_, 'r') as f:
        text = f.read()
    
    if '\"\"\"' not in text:
        if sample_indices:
            sample_indices.sort()
            with open(file_, 'r') as f:
                return [next(f).rstrip() for _ in range(max(sample_indices)+1) if _ in sample_indices]
        else:
            return [line.rstrip() for line in text.splitlines()]
    else:
        ret = [txt.strip() for txt in text.split("\"\"\"")[1::2]]
        if sample_indices:
            return [ret[i] for i in sample_indices]
        else:
            return ret

def k_init_pop(initial_mode: str, init_population: List[str], k: int) -> List[str]:
    """Initialize population with k individuals"""
    if initial_mode == "topk":
        return init_population[:k]
    elif initial_mode == "para_topk":
        return init_population[:k // 2]
    elif initial_mode == "para_bottomk":
        return init_population[-k // 2:]
    elif initial_mode == "para_randomk":
        return random.sample(init_population, k // 2)
    elif initial_mode == "randomk":
        return random.sample(init_population, k)
    elif initial_mode == "bottomk":
        return init_population[-k:]
    else:
        raise ValueError(f"Invalid initial mode: {initial_mode}")

def examples2string(examples_dict: Dict[str, Any]) -> str:
    """
    Convert examples dictionary returned by WrongExampleCollector to formatted string
    """
    input_list = examples_dict["input_list"]
    wrong_label_list = examples_dict["wrong_label_list"]
    wrong_annot_list = examples_dict["wrong_annot_list"]
    true_label_list = examples_dict["true_label_list"]
    true_annot_list = examples_dict["true_annot_list"]
    
    ret_str = ""
    for i in range(len(input_list)):
        ret_str += f"Case <{i}>\n"
        ret_str += f"Model input: {input_list[i]}\n\n"
        ret_str += f"True label: {true_label_list[i]}\n"
        ret_str += f"Model prediction: {wrong_label_list[i]}\n\n"
        ret_str += f"True annotation: {true_annot_list[i]}\n"
        ret_str += f"Model annotation: {wrong_annot_list[i]}\n\n"
        
    return ret_str


class WrongExampleCollector:
    """Collect wrong examples from model outputs"""
    
    def __init__(self, n_max: int = 3, method: str = "random",
                 topic_finder: Optional[Any] = None):
        """
        Initialize wrong example collector.
        
        Args:
            n_max: Maximum number of examples to collect per category
            method: Collection method ("random" or "topic")
            topic_finder: Optional topic finder instance for topic-based collection
        """
        self.n_max = n_max
        self.method = method
        if method not in ["random", "topic"]:
            raise ValueError("Invalid collecting method!")
            
        self.topic_finder = topic_finder

    def forward(self,
                input_list: List[str],
                true_labels: List[int],
                true_annots: List[str],
                model_decisions: List[str],
                model_annotations: List[str]) -> Dict[str, Any]:
        """
        Select wrong examples from model outputs.
        
        Args:
            input_list: List of order inputs
            true_labels: List of true labels
            true_annots: List of true annotations
            model_decisions: List of model decisions
            model_annotations: List of model annotations
            
        Returns:
            Dictionary containing collected examples
        """
        assert len(true_annots) == len(model_annotations), "Invalid input!"
        
        # # Convert model decisions to labels
        # model_labels = [0 if dec == "PASS" else 1 for dec in model_decisions]
        # true_decisions = ["PASS" if lab == 0 else "ENFORCE" for lab in true_labels]
        
        # Find indices of wrong predictions
        # wrong0_true1_idx = np.where((np.array(model_labels) == 0) * (np.array(true_labels) == 1))[0]
        # wrong1_true0_idx = np.where((np.array(model_labels) == 1) * (np.array(true_labels) == 0))[0]
        wrong_idx = np.where(np.array([m != t for m, t in zip(model_decisions, true_labels)]))[0]
        
        if self.method == "random":
            all_idx = self._collect_random(wrong_idx)
        elif self.method == "topic":
            all_idx = self.topic_finder.forward(true_labels, true_annots, model_decisions)
        else:
            raise NotImplementedError
            
        collected_example_dict = {
            "input_list": [input_list[idx] for idx in all_idx],
            "wrong_label_list": [model_decisions[idx] for idx in all_idx],
            "wrong_annot_list": [model_annotations[idx] for idx in all_idx],
            "true_label_list": [true_labels[idx] for idx in all_idx],
            "true_annot_list": [true_annots[idx] for idx in all_idx],
            "all_idx": all_idx,
        }
        
        print(f"Found {len(all_idx)} wrong examples")
        return collected_example_dict

    def _collect_random(self, wrong_idx: np.ndarray) -> List[int]:
        """Collect examples randomly"""
        idx_set = (list(wrong_idx[
            np.random.choice(
                np.arange(len(wrong_idx)),
                size=self.n_max,
                replace=False,
            )
        ]) if len(wrong_idx) > self.n_max 
            else [i for i in wrong_idx])
        
        return sorted(idx_set)


class KmeansTopicCluster:
    """Find wrong sample clusters using K-means"""
    
    def __init__(self,
                 embedding_model: Any,
                 n_case: int = 10,
                 n_case_to_cluster: int = 100,
                 nc_min: int = 5,
                 nc_max: int = 20):
        """
        Initialize KMeans topic cluster.
        
        Args:
            embedding_model: Model for generating embeddings
            n_case: Number of cases to return
            n_case_to_cluster: Number of cases to consider for clustering
            nc_min: Minimum number of clusters
            nc_max: Maximum number of clusters
        """
        self.n_case = n_case
        self.n_case_to_cluster = n_case_to_cluster
        self.nc_min = nc_min
        self.nc_max = nc_max
        self.embedding_model = embedding_model

    def _choose_best_k(self, scaled_data: np.ndarray,
                      k_range: List[int],
                      alpha_k: float = 0.02) -> Tuple[int, np.ndarray]:
        """
        Choose the optimal number of clusters.
        
        Args:
            scaled_data: Data to cluster
            k_range: Range of k values to try
            alpha_k: Penalty factor for number of clusters
            
        Returns:
            Tuple of (best k, results array)
        """
        ans = []
        inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
            scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
            ans.append((k, scaled_inertia))
            
        results = np.array(ans)
        best_k = results[results[:, 1].argmin(), 0]
        return int(best_k), results

    def forward(self,
                true_labels: List[int],
                true_annots: List[str],
                model_decisions: List[float]) -> List[int]:
        """
        Find clusters of wrong examples.
        
        Args:
            true_labels: List of true labels
            true_annots: List of true annotations
            model_decisions: List of model decisions
            
        Returns:
            List of selected indices
        """
        # Find wrong predictions
        wrong_idx = np.where(np.array([m != t for m, t in zip(model_decisions, true_labels)]))[0]
        n_wrong = len(wrong_idx)
        n_select = min(self.n_case_to_cluster, n_wrong)
        
        # Select n_select cases from wrong_idx
        if n_wrong > n_select:
            idx_to_cluster = np.random.choice(
                wrong_idx,
                size=n_select,
                replace=False,
            )
        else:
            idx_to_cluster = wrong_idx
        true_annots_to_cluster = np.array(true_annots)[idx_to_cluster]
        
        # Generate embeddings
        embeddings = np.array([
            self.embedding_model.encode(annot)
            for annot in true_annots_to_cluster
        ])
        
        if n_select <= 1:
            return idx_to_cluster

        # Choose optimal number of clusters
        k_range = list(range(
            min(n_select, self.nc_min),
            min(n_select, self.nc_max) + 1
        ))
        opt_k, _ = self._choose_best_k(embeddings, k_range)
        
        print(f"Found {opt_k} clusters")
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=opt_k,
            max_iter=200,
            n_init=1,
            random_state=10,
        ).fit(embeddings)
        
        # Select examples from largest cluster
        labels = kmeans.labels_
        v, c = np.unique(labels, return_counts=True)
        ret_v = v[np.argmax(c)]
        
        ret_idx = idx_to_cluster[np.where(labels == ret_v)[0]][:self.n_case]
        
        return ret_idx