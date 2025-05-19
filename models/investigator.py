# GreenTEA/models/investigator.py

import json
import boto3
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

class LLMInvestigator:
    """
    LLM-based investigator.
    """
    def __init__(self,
                 prompt_template_path: str,
                 model_name: str,
                 system_prompt_path: Optional[str] = None,
                 task_prompt_path: Optional[str] = None):
        """
        Initialize LLM investigator.
        
        Args:
            prompt_template_path: Path to prompt template file
            model_name: Name of the LLM model to use
            system_prompt_path: Optional path to system prompt file
            task_prompt_path: Optional path to task prompt file
        """
        self.model_name = model_name
        self.client = boto3.client('bedrock-runtime')
        
        # Load prompt template
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read()
            
        # Load system prompt if provided
        self.system_prompt = None
        if system_prompt_path:
            with open(system_prompt_path, 'r') as f:
                self.system_prompt = f.read()
                
        # Load task prompt if provided
        self.task_prompt = None
        if task_prompt_path:
            with open(task_prompt_path, 'r') as f:
                self.task_prompt = f.read()

    def _extract_seconds(self, text: str) -> int:
        """Extract number of seconds from error message"""
        words = text.split()
        for i, word in enumerate(words):
            if "second" in word:
                return int(words[i - 1])
        return 20

    def _call_llm(self,
                  prompt_data: str,
                  temperature: float = 0.0,
                  max_tokens: int = 2000,
                  top_p: float = 1.0,
                  top_k: int = 1,
                  max_retries: int = 2) -> str:
        """
        Call LLM with retry logic.
        
        Args:
            prompt_data: Input prompt for the model
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_retries: Maximum number of retries
            
        Returns:
            Model response text
        """
        retried = 0
        while retried < max_retries:
            try:
                temp_client = boto3.client('bedrock-runtime')
                
                if "claude" in self.model_name:
                    # Construct Claude request payload
                    message_content = [{
                        "role": "user",
                        "content": [{"type": "text", "text": prompt_data}]
                    }]
                    json_dict = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "max_tokens": max_tokens,
                        "messages": message_content,
                    }
                    if self.system_prompt:
                        json_dict["system"] = self.system_prompt
                    
                    # Invoke Claude model
                    response = temp_client.invoke_model(
                        modelId=self.model_name,
                        body=json.dumps(json_dict),
                    )
                    response_body = json.loads(response.get("body").read())
                    response_text = response_body["content"][0]["text"]

                elif "nova" in self.model_name:
                    # Construct Titan Nova request payload
                    message_content = [{"role": "user", "content": [{"text": prompt_data}]}]
                    json_dict = {
                        "messages": message_content,
                        "inferenceConfig": {
                            "maxTokens": max_tokens,
                            "temperature": temperature,
                            "topP": top_p,
                            "topK": top_k
                        }
                    }
                    if self.system_prompt:
                        json_dict["system"] = [{"text": self.system_prompt}]
                    
                    # Invoke Nova model
                    response = temp_client.invoke_model(
                        modelId=self.model_name,
                        body=json.dumps(json_dict)
                    )
                    response_body = json.loads(response.get("body").read())
                    response_text = response_body['output']['message']['content'][0]['text']

                else:
                    raise ValueError(f"Unsupported model: {self.model_name}")
                    
                return response_text

            except Exception as e:
                error = str(e)
                logging.debug(f"LLM call failed: {error}")
                retried += 1
                # Extract retry delay and apply it
                second = self._extract_seconds(error)
                time.sleep(second)

        raise RuntimeError(f"LLM call failed after {max_retries} attempts.")

    def _process_batch(self,
                      input_chunk: List[str],
                      task_prompt: Optional[str] = None,
                      system_prompt: Optional[str] = None,
                      **kwargs) -> List[str]:
        """Process a batch of orders"""
        if task_prompt:
            prompt_text = task_prompt
        elif system_prompt:
            prompt_text = system_prompt
        else:
            prompt_text = self.task_prompt

        return [
            self._call_llm(
                self.prompt_template.format(
                    input_text=model_input,
                    adjustable_prompt=prompt_text
                ),
                **kwargs
            )
            for model_input in input_chunk
        ]

    def _split_list(self, lst: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split a list into chunks"""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def forward(self,
                input_text_list: List[str],
                task_prompt: Optional[str] = None,
                system_prompt: Optional[str] = None,
                chunk_size: int = 50,
                num_workers: int = 4,
                **kwargs) -> List[str]:
        """
        Process multiple orders in parallel.
        
        Args:
            input_text_list: List of model input texts
            task_prompt: Optional task-specific prompt
            system_prompt: Optional system prompt
            chunk_size: Size of batches for parallel processing
            num_workers: Number of parallel workers
            **kwargs: Additional arguments for LLM call
            
        Returns:
            List of model responses
        """
        # Split inputs into chunks
        input_chunks = self._split_list(input_text_list, chunk_size)

        results = [None] * len(input_chunks)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Initialize progress bar
            with tqdm(total=len(input_chunks), desc="Processing orders") as pbar:
                # Submit tasks
                futures = {
                    executor.submit(
                        self._process_batch,
                        input_chunks[i],
                        task_prompt,
                        system_prompt,
                        **kwargs
                    ): i
                    for i in range(len(input_chunks))
                }
                
                # Process results as they complete
                for future in as_completed(futures):
                    index = futures[future]
                    results[index] = future.result()
                    pbar.update()

        # Flatten results
        return [item for sublist in results if sublist is not None for item in sublist]
