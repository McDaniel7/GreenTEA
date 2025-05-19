# GreenTEA/utils/ga_utils.py

import os
import re
import json
import random
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ast import literal_eval
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .text_utils import parse_tagged_text, examples2string, KmeansTopicCluster, WrongExampleCollector

class ParentPromptSelectorWheel:
    """Roulette wheel parent selector"""
    
    def __init__(self, pop_size: int):
        """
        Initialize wheel selector.
        
        Args:
            pop_size: Size of population
        """
        self.pop_size = pop_size
        
    def forward(self, prompts: List[str], scores: np.ndarray) -> Tuple[str, str]:
        """
        Select two parent prompts using roulette wheel selection.
        
        Args:
            prompts: List of prompt candidates
            scores: Array of corresponding scores
            
        Returns:
            Tuple of selected parent prompts
        """
        wheel_idx = np.random.choice(
            np.arange(len(prompts)),
            size=2,
            replace=True,
            p=scores / scores.sum(),
        ).tolist()
        
        temp_parent_pop = [p for p in prompts]
        return temp_parent_pop[wheel_idx[0]], temp_parent_pop[wheel_idx[1]]

class ParentPromptSelectorRandom:
    """Random parent selector"""
    
    def __init__(self, pop_size: int):
        """
        Initialize random selector.
        
        Args:
            pop_size: Size of population
        """
        self.pop_size = pop_size
        
    def forward(self, prompts: List[str], scores: np.ndarray) -> Tuple[str, str]:
        """
        Select two parent prompts randomly.
        
        Args:
            prompts: List of prompt candidates
            scores: Array of corresponding scores (not used in random selection)
            
        Returns:
            Tuple of selected parent prompts
        """
        temp_parent_pop = [p for p in prompts]
        parents = random.sample(temp_parent_pop, 2)
        return parents[0], parents[1]

class Evolutor:
    """Genetic algorithm evolutor"""
    
    def __init__(self,
                 config: Dict[str, Any],
                 data_preparer: Any,
                 investigator: Any,
                 evaluator: Any,
                 prompt_generator: Any,
                 prompt_paraphraser: Any,
                 error_hunter: Optional[Any] = None):
        """
        Initialize evolutor.
        
        Args:
            config: Configuration dictionary
            data_preparer: Data preparation module
            investigator: LLM investigator module
            evaluator: Evaluation module
            prompt_generator: Prompt generation module
            prompt_paraphraser: Prompt paraphrasing module
            error_hunter: Optional error hunting module
        """
        self.config = config
        self.data_preparer = data_preparer
        self.investigator = investigator
        self.evaluator = evaluator
        self.child_prompt_generator = prompt_generator
        self.prompt_paraphraser = prompt_paraphraser
        
        # Set up parent selector
        if config["parent_selection"] == "wheel":
            self.parent_selector = ParentPromptSelectorWheel(config["pop_size"])
        elif config["parent_selection"] == "random":
            self.parent_selector = ParentPromptSelectorRandom(config["pop_size"])
        else:
            raise ValueError("Invalid parent selection method!")
        
        # Set up gradient guidance if enabled
        if config["gradient_guided"]:
            self.gradient_guided = True
            self.wrong_example_collector = self._setup_wrong_example_collector(config)
            self.error_hunter = error_hunter
        else:
            self.gradient_guided = False
        
        # Initialize storage
        self.init_population = []
        self.population = []
        self.scores = []
        self.marks = []
        
        # Set up logging and output paths
        self.public_out_path = config["output_path"]
        if not os.path.exists(self.public_out_path):
            os.makedirs(self.public_out_path)
            
        self.logger = self._setup_logging()

    def _setup_wrong_example_collector(self, config: Dict[str, Any]) -> Any:
        """Setup wrong example collector based on configuration"""
        
        sbert_cluster = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        if config["exp_collect_mode"] == 'topic':
            topic_finder = KmeansTopicCluster(
                sbert_cluster,
                n_case=config["exp_collect_n_max"],
                n_case_to_cluster=100,
                nc_min=5,
                nc_max=20
            )
            return WrongExampleCollector(
                config["exp_collect_n_max"],
                config["exp_collect_mode"],
                topic_finder=topic_finder
            )
        else:
            return WrongExampleCollector(
                config["exp_collect_n_max"],
                config["exp_collect_mode"]
            )

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(self.config["logger_name"])
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(
                os.path.join(self.public_out_path, f"evol.log")
            )
            formatter = logging.Formatter("[%(asctime)s] - %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    def _prepare_evaluation_data(self):
        """Prepare development and test data"""
        self.dev_eval_src = self.data_preparer.forward(
            self.dev_data_df,
            desc="Processing dev data"
        )
        
        if self.config["testing"]:
            self.test_eval_src = self.data_preparer.forward(
                self.test_data_df,
                desc="Processing test data"
            )

    def sorting(self):
        """Sort population by scores"""
        best_score = 0
        total_score = 0
        
        with open(os.path.join(self.public_out_path, "dev_result.txt"), "w") as wf:
            self.scores, self.population, self.marks = (
                list(t) for t in zip(*sorted(
                    zip(self.scores, self.population, self.marks),
                    key=lambda x: x[0],
                    reverse=True
                ))
            )
            
            for score, prompt, mark in zip(self.scores, self.population, self.marks):
                score_str = "[" + ",".join([str(round(i, 4)) for i in score]) + "]"
                float_score = float(score[-1])
                if float_score > best_score:
                    best_score = float_score
                total_score += float_score
                
                if self.gradient_guided:
                    wf.write(
                        f"{mark}\t{prompt}\t{score_str}\t"
                        f"{self.evaluated_prompts[prompt]['wrong_examples']}\t"
                        f"{self.evaluated_prompts[prompt]['hunted_error']}\n"
                    )
                else:
                    wf.write(f"{mark}\t{prompt}\t{score_str}\n")
                    
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {total_score / len(self.scores)}\n")

    def write_step(self, step: int, file_type: str, sorted_population: list):
        """Write step results to file"""

        best_score = self.evaluated_prompts[sorted_population[0]]["scores"][-1]
        avg_score = sum([self.evaluated_prompts[p]["scores"][-1] for p in sorted_population]) / len(sorted_population)
        
        with open(os.path.join(
            self.public_out_path,
            f"step{step}_{file_type}_{self.config['save_file_name']}.txt"
        ), "w") as wf:
            for p in sorted_population:
                score_str = "[" + ",".join(
                    [str(round(i, 4)) for i in self.evaluated_prompts[p]["scores"]]
                ) + "]"
                if self.gradient_guided:
                    wf.write(
                        f"{self.prompts2mark[p]}\t{p}\t{score_str}\t"
                        f"{self.evaluated_prompts[p]['wrong_examples']}\t"
                        f"{self.evaluated_prompts[p]['hunted_error']}\n"
                    )
                else:
                    wf.write(f"{self.prompts2mark[p]}\t{p}\t{score_str}\n")
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {avg_score}\n")

    def eval_single_prompt(self, prompt: str,
                           task_prompt: bool = True,
                           dev: bool = True,
                           **kwargs) -> Dict[str, Any]:
        """Evaluate a single prompt"""
        # Prepare data
        if dev:
            input_list_total, true_annots_total, true_labels_total = self.dev_eval_src
            while True:
                rnd_idx = np.random.choice(list(range(len(input_list_total))), size=self.config["batch_size"], replace=False)
                input_list = [input_list_total[idx] for idx in rnd_idx]
                true_annots = [true_annots_total[idx] for idx in rnd_idx]
                true_labels = [true_labels_total[idx] for idx in rnd_idx]
                if len(np.unique(true_labels)) > 1:
                    break
        else:
            input_list, true_annots, true_labels = self.test_eval_src

        # Model inference
        num_workers = 4
        chunk_size = int(np.ceil(len(input_list)/num_workers))
        if task_prompt:
            model_response = self.investigator.forward(
                input_list, task_prompt=prompt, chunk_size=chunk_size, num_workers=num_workers, **kwargs
            )
        else:
            model_response = self.investigator.forward(
                input_list, system_prompt=prompt, chunk_size=chunk_size, num_workers=num_workers, **kwargs
            )

        # Parse responses
        model_decisions = [parse_tagged_text(res, "<Decision>", "</Decision>") for res in model_response]
        model_annotations = [parse_tagged_text(res, "<Annotation>", "</Annotation>") for res in model_response]
        
        valid_idx = np.where((np.array(model_decisions) != "") * (np.array(model_annotations) != ""))[0]
        
        input_list = [input_list[idx] for idx in valid_idx]
        true_annots = [true_annots[idx] for idx in valid_idx]
        true_labels = [true_labels[idx] for idx in valid_idx]
        model_decisions = [model_decisions[idx] for idx in valid_idx]
        model_annotations = [model_annotations[idx] for idx in valid_idx]

        if len(model_annotations) == 0:
            model_score = [0.0]
            collected_wrong_example_dict = {"all_idx": []}
            wrong_examples_string = "Invalid wrong examples"
            hunted_error = "Invalid hunted error"
        else:
            # Evaluation
            model_score = self.evaluator.forward(model_decisions, true_labels)
        
            if self.gradient_guided:
                collected_wrong_example_dict = self.wrong_example_collector.forward(
                    input_list, true_labels, true_annots,
                    model_decisions, model_annotations
                )
                wrong_examples_string = examples2string(collected_wrong_example_dict)
                hunted_error = self.error_hunter.forward(
                    cur_prompt=prompt, wrong_examples=wrong_examples_string,
                    **self.config["error_hunter_config"]
                )

        if self.gradient_guided:
            return {
                "true_annots": true_annots,
                "true_labels": true_labels,
                "full_responses": model_response,
                "scores": model_score,
                "wrong_examples_string": wrong_examples_string,
                "wrong_examples_idx": str(collected_wrong_example_dict["all_idx"]),
                "hunted_error": hunted_error,
            }
        else:
            return {
                "true_annots": true_annots,
                "true_labels": true_labels,
                "full_responses": model_response,
                "scores": model_score,
            }

    def init_pop(self, 
                 manual_prompt_file_path: Optional[str] = None,
                 llm_prompt_file_path: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        """Initialize population"""
        # Load and prepare evaluation data
        self._prepare_evaluation_data()
        
        config = self.config
        logger = self.logger

        prompts2mark = {}
        manual_prompt_path = manual_prompt_file_path if manual_prompt_file_path else config["manual_prompt_file_path"]
        llm_prompt_path = llm_prompt_file_path if llm_prompt_file_path else config["llm_prompt_file_path"]

        manual_pop = self._read_lines(manual_prompt_path)
        try:
            llm_pop = self._read_lines(llm_prompt_path)
        except:
            llm_pop = []
        
        for p in llm_pop:
            prompts2mark[p] = "llm"
        for p in manual_pop:
            prompts2mark[p] = "manual"

        self.evaluated_prompts = {}
        out_path = self.public_out_path
        cur_budget = -1

        if config["init_mode"] == "all":
            cache_path = self._get_cache_path()
            
            try:
                self.evaluated_prompts = json.load(open(cache_path, "r"))
                logger.info(f"---loading prompts from {cache_path}")
                metric_index = -1
                self.evaluated_prompts = dict(sorted(
                    self.evaluated_prompts.items(),
                    key=lambda item: item[1]["scores"][metric_index],
                    reverse=True
                ))
                init_population = list(self.evaluated_prompts.keys())
            except:
                topk_population = []
                logger.info("-----evaluating initial population and paraphrasing topk---------")
                for prompt in manual_pop + llm_pop:
                    eval_res = self.eval_single_prompt(
                        prompt, task_prompt=config["task_prompt"],
                        dev=True, **config["investigator_config"]
                    )
                    scores = eval_res["scores"]
                    self.evaluated_prompts[prompt] = {"scores": scores}
                    if self.gradient_guided:
                        self.evaluated_prompts[prompt]["wrong_examples"] = eval_res["wrong_examples_string"]
                        self.evaluated_prompts[prompt]["hunted_error"] = eval_res["hunted_error"]
                        self.evaluated_prompts[prompt]["wrong_examples_idx"] = eval_res["wrong_examples_idx"]
                    topk_population.append((scores[-1], prompt))
                topk_population.sort(reverse=True, key=lambda x: x[0])
                
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w") as wf:
                    self.evaluated_prompts = dict(sorted(
                        self.evaluated_prompts.items(),
                        key=lambda item: item[1]["scores"][-1],
                        reverse=True
                    ))
                    json.dump(self.evaluated_prompts, wf)
                init_population = [i[1] for i in topk_population]

        elif config["init_mode"] == "ckpt":
            init_population = []
            logger.info(f"------------load from file {config['ckpt_pop']}------------")
            ckpt_pop = self._read_ckpt_files(config["ckpt_pop"])[:config["pop_size"]]
            for line in ckpt_pop:
                try:
                    elements = line.split("\t")
                    mark, prompt = elements[0], elements[1]
                    score = literal_eval(elements[2])
                    score = [float(i) for i in score]
                    if self.gradient_guided:
                        wrong_examples = elements[3]
                        hunted_error = elements[4]
                except:
                    continue
                prompts2mark[prompt] = mark
                self.evaluated_prompts[prompt] = {"scores": [i for i in score]}
                init_population.append(prompt)
                if self.gradient_guided:
                    self.evaluated_prompts[prompt]["wrong_examples"] = wrong_examples
                    self.evaluated_prompts[prompt]["hunted_error"] = hunted_error
            cur_budget = self._extract_numbers(config["ckpt_pop"].split("/")[-1])
            logger.info(f"cur budget is {cur_budget}")

        if config["init_pop"] in ["para_topk", "para_bottomk", "para_randomk"]:
            k_pop = self._k_init_pop(config["init_pop"], init_population, k=config["pop_size"])
            logger.info("-----paraphrasing---------")
            para_population = self.prompt_paraphraser.forward(
                sentence=k_pop, **config["paraphraser_config"]
            )
            for p in para_population:
                prompts2mark[p] = "para"
                eval_results = self.eval_single_prompt(
                    p, task_prompt=config["task_prompt"],
                    dev=True, **config["investigator_config"]
                )
                self.evaluated_prompts[p] = {"scores": eval_results["scores"]}
                if self.gradient_guided:
                    self.evaluated_prompts[p]["wrong_examples"] = eval_results["wrong_examples_string"]
                    self.evaluated_prompts[p]["hunted_error"] = eval_results["hunted_error"]
                    self.evaluated_prompts[p]["wrong_examples_idx"] = eval_results["wrong_examples_idx"]
            init_population = k_pop + para_population
            init_population = init_population[:config["pop_size"]]
        elif config["init_pop"] in ["topk", "bottomk", "randomk"]:
            init_population = self._k_init_pop(
                config["init_pop"], init_population, k=config["pop_size"]
            )

        self.population = [i for i in init_population]
        assert len(self.population) <= config["pop_size"]

        for i in self.population:
            logger.info(i)
            
        if config["init_mode"] != "ckpt":
            with open(os.path.join(out_path, f"step-1_pop_{config['save_file_name']}.txt"), "w") as wf:
                for prompt in self.population:
                    score_str = "[" + ",".join(
                        [str(round(i, 4)) for i in self.evaluated_prompts[prompt]["scores"]]
                    ) + "]"
                    if self.gradient_guided:
                        wf.write(f"{prompts2mark[prompt]}\t{prompt}\t{score_str}\t{self.evaluated_prompts[prompt]['wrong_examples']}\t{self.evaluated_prompts[prompt]['hunted_error']}\n")
                    else:
                        wf.write(f"{prompts2mark[prompt]}\t{prompt}\t{score_str}\n")

        self.prompts2mark = prompts2mark
        return self.evaluated_prompts, cur_budget

    def _run_testing_phase(self, step: int) -> None:
        """
        Run testing phase on the best prompts.
        
        Args:
            step: Current evolution step
        """
        logger = self.logger
        logger.info(f"----------testing step {step} population----------")
        
        # Sort population by their development scores
        pop_marks = [self.prompts2mark[i] for i in self.population]
        pop_scores = [self.evaluated_prompts[i]["scores"] for i in self.population]
        self.population, pop_scores, pop_marks = (
            list(t) for t in zip(
                *sorted(
                    zip(self.population, pop_scores, pop_marks),
                    key=lambda x: x[1][-1],
                    reverse=True
                )
            )
        )
    
        # Test top N prompts
        test_prompt_num = min(3, len(self.population))
        marks = []
        prompts = []
        all_scores = []
        scores_strs = []
        
        with open(os.path.join(self.public_out_path, f"step{step}_pop_test_{self.config['save_file_name']}.txt"), "w") as wf:
            # Evaluate each of the top prompts
            for prompt, mark in zip(self.population[:test_prompt_num], pop_marks[:test_prompt_num]):
                eval_test_res = self.eval_single_prompt(
                    prompt,
                    dev=False,
                    task_prompt=self.config["task_prompt"],
                    **self.config["investigator_config"]
                )
                scores = eval_test_res["scores"]
                all_scores.append(scores[-1])
                
                score_str = "[" + ",".join(
                    [str(round(i, 4)) for i in scores]
                ) + "]"
                
                if self.gradient_guided:
                    wf.write(
                        f"{mark}\t{prompt}\t{score_str}\t"
                        f"{eval_test_res['wrong_examples_string']}\t"
                        f"{eval_test_res['hunted_error']}\n"
                    )
                else:
                    wf.write(f"{mark}\t{prompt}\t{score_str}\n")
                    
                scores_strs.append(score_str)
                marks.append(mark)
                prompts.append(prompt)
                wf.flush()
    
            # Sort results by test scores
            score_sorted, prompts_sorted, mark_sorted, scores_strs_sorted = (
                list(t) for t in zip(
                    *sorted(
                        zip(all_scores, prompts, marks, scores_strs),
                        reverse=True
                    )
                )
            )
    
            # Write sorted results
            wf.write("\n----------sorted results----------\n")
            for i in range(len(score_sorted)):
                wf.write(
                    f"{mark_sorted[i]}\t{prompts_sorted[i]}\t{scores_strs_sorted[i]}\n"
                )
            
            best_score = score_sorted[0]
            best_prompt = prompts_sorted[0]
        
        logger.info(
            f"----------Testing step {step} best score: {best_score}, best prompt: {best_prompt}----------"
        )

    def _read_lines(self, file_: str, sample_indices: Optional[List[int]] = None) -> List[str]:
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

    def _read_ckpt_files(self, file_path: str) -> List[str]:
        """Read checkpoint files"""
        with open(file_path, "r") as f:
            all_text = f.read()
        
        delimiters_list = ["manual\t", "evoluted\t", "llm\t", "best score: "]
        splitted = re.split('|'.join(map(re.escape, delimiters_list)), all_text)
        delimits = re.findall('|'.join(map(re.escape, delimiters_list)), all_text)
        
        splitted = splitted[1:-1]
        delimits = delimits[:-1]
        
        assert len(splitted) == len(delimits)
        return [d + s for d, s in zip(delimits, splitted)]

    def _k_init_pop(self, initial_mode: str, init_population: List[str], k: int) -> List[str]:
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

    def _get_cache_path(self) -> str:
        """Get cache path for evaluated prompts"""
        config = self.config
        if "haiku" in config["investigator_model_name"]:
            simple_model_name = "claude-3-5-haiku"
        elif "sonnet" in config["investigator_model_name"]:
            simple_model_name = "claude-3-sonnet"
        elif "gpt" in config["investigator_model_name"]:
            simple_model_name = "gpt"
        else:
            simple_model_name = "unknown_model"
        
        return (config["cache_path"] if config["cache_path"] else
                f"{config['rootpath']}project_data/GA_data/cache/{config['task']}/{config['dev_file']}/seed_{config['seed']}/prompts_{simple_model_name}.json")

    def evolute(self, 
                manual_prompt_file_path: Optional[str] = None,
                llm_prompt_file_path: Optional[str] = None):
        """
        Run the evolution process.
        
        Args:
            manual_prompt_file_path: Path to manual prompts file
            llm_prompt_file_path: Path to LLM-generated prompts file
        """
        logger = self.logger
        self.evaluated_prompts, cur_budget = self.init_pop(
            manual_prompt_file_path=manual_prompt_file_path,
            llm_prompt_file_path=llm_prompt_file_path
        )
        
        best_scores = []
        avg_scores = []
        
        # Evolution loop
        for step in range(cur_budget + 1, self.config["budget"]):
            total_score = 0
            best_score = 0
            new_pop = []
            
            parent_pop = [p for p in self.population]
            scores = np.array([
                self.evaluated_prompts[p]["scores"][-1]
                for p in self.population
            ])

            # Generate new population
            for j in range(self.config["pop_size"]):
                logger.info(f"step {step}, new pop {j}")
                
                # Select parents and generate child
                prompt1, prompt2 = self.parent_selector.forward(parent_pop, scores)
                logger.info("Selected parents:")
                logger.info(prompt1)
                logger.info(prompt2)
                
                if self.gradient_guided:
                    child_prompt = self.child_prompt_generator.forward(
                        prompt1,
                        self.evaluated_prompts[prompt1]["wrong_examples"],
                        self.evaluated_prompts[prompt1]["hunted_error"],
                        prompt2,
                        self.evaluated_prompts[prompt2]["wrong_examples"],
                        self.evaluated_prompts[prompt2]["hunted_error"],
                        **self.config["child_generator_config"]
                    )
                else:
                    child_prompt = self.child_prompt_generator.forward(
                        prompt1,
                        prompt2,
                        **self.config["child_generator_config"]
                    )
                logger.info(f"Child prompt: {child_prompt}")

                # Evaluate child
                ga_eval_res = self.eval_single_prompt(
                    child_prompt,
                    dev=True,
                    task_prompt=self.config["task_prompt"],
                    **self.config["investigator_config"]
                )
                
                ga_scores = ga_eval_res["scores"]
                ga_score_str = "\t".join([str(round(i, 4)) for i in ga_scores])
                new_score = ga_scores[-1]

                logger.info(f"New score: {ga_score_str}")
                
                # Store results
                self.prompts2mark[child_prompt] = "evoluted"
                self.evaluated_prompts[child_prompt] = {"scores": ga_scores}
                if self.gradient_guided:
                    self.evaluated_prompts[child_prompt].update({
                        "wrong_examples": ga_eval_res["wrong_examples_string"],
                        "hunted_error": ga_eval_res["hunted_error"],
                        "wrong_examples_idx": ga_eval_res["wrong_examples_idx"]
                    })

                new_pop.append(child_prompt)
                total_score += new_score
                if new_score > best_score:
                    best_score = new_score

            # write new population
            sorted_new_pop = sorted(
                new_pop,
                key=lambda x: self.evaluated_prompts[x]["scores"][-1],
                reverse=True,
            )
            self.write_step(step, 'new-pop', sorted_new_pop)

            # Update population
            double_pop = list(set(self.population + new_pop))
            double_pop = sorted(
                double_pop,
                key=lambda x: self.evaluated_prompts[x]["scores"][-1],
                reverse=True
            )
            self.population = double_pop[:self.config["pop_size"]]
            
            # Calculate and store metrics
            total_score = sum(
                self.evaluated_prompts[i]["scores"][-1]
                for i in self.population
            )
            best_score = self.evaluated_prompts[self.population[0]]["scores"][-1]
            avg_score = total_score / self.config["pop_size"]
            
            avg_scores.append(avg_score)
            best_scores.append(best_score)

            self.write_step(step, 'pop', self.population)

            # Test phase if needed
            if step == self.config["budget"] - 1 and self.config["test_file"] != "NA":
                self._run_testing_phase(step)

        # Log final results
        best_scores = [str(i) for i in best_scores]
        avg_scores = [str(round(i, 4)) for i in avg_scores]
        logger.info(f"best_scores: {','.join(best_scores)}")
        logger.info(f"avg_scores: {','.join(avg_scores)}")
        
        self.scores = [self.evaluated_prompts[i]["scores"] for i in self.population]
        self.marks = [self.prompts2mark[i] for i in self.population]
        self.sorting()

    @staticmethod
    def _extract_numbers(string: str) -> int:
        """Extract numbers from string"""
        return [int(num) for num in re.findall(r'\d+', string)][0]