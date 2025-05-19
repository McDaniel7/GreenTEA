# GreenTEA/models/prompt_generator.py

import boto3
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

class LLMBaseGenerator:
    """Base class for LLM-based generators"""
    
    def __init__(self, prompt_template_path: str, model_name: str):
        """
        Initialize base generator.
        
        Args:
            prompt_template_path: Path to prompt template file
            model_name: Name of the LLM model to use
        """
        self.model_name = model_name
        self.client = boto3.client('bedrock-runtime')
        
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read()

    def _extract_seconds(self, text: str) -> int:
        """Extract number of seconds from error message"""
        words = text.split()
        for i, word in enumerate(words):
            if "second" in word:
                return int(words[i - 1])
        return 20

    def _call_llm(self,
                  prompt_data: str,
                  temperature: float = 0.5,
                  max_tokens: int = 1000,
                  top_p: float = 1.0,
                  top_k: int = 250,
                  max_retries: int = 2) -> str:
        """Base LLM calling method with retry logic"""
        retried = 0
        while retried < max_retries:
            try:
                temp_client = boto3.client('bedrock-runtime')
                
                if "claude" in self.model_name:
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
                    
                    response = temp_client.invoke_model(
                        modelId=self.model_name,
                        body=json.dumps(json_dict),
                    )
                    response_body = json.loads(response.get("body").read())
                    response_text = response_body["content"][0]["text"]
                    
                elif "nova" in self.model_name:
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
                second = self._extract_seconds(error)
                time.sleep(second)

        raise RuntimeError(f"LLM call failed after {max_retries} attempts.")

    def _parse_tagged_text(self, text: str, start_tag: str, end_tag: str) -> str:
        """Extract text between tags"""
        try:
            start_index = text.find(start_tag)
            if start_index == -1:
                return ""
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                return ""
            return text[start_index + len(start_tag):end_index].strip()
        except Exception as e:
            logging.error(f"Error parsing tagged text: {e}")
            return ""

class LLMChildGenerator(LLMBaseGenerator):
    """LLM-based child prompt generator"""
    
    def forward(self,
                prompt1: str,
                prompt2: str,
                **kwargs) -> str:
        """
        Generate a child prompt from two parent prompts.
        
        Args:
            prompt1: First parent prompt
            prompt2: Second parent prompt
            **kwargs: Additional arguments for LLM call
            
        Returns:
            Generated child prompt
        """
        prompt_data = self.prompt_template.format(prompt1=prompt1, prompt2=prompt2)
        new_prompt = self._call_llm(prompt_data, **kwargs)
        return self._parse_tagged_text(new_prompt, "<NewPrompt>", "</NewPrompt>")

class LLMGuidedCrossoverChildGenerator(LLMBaseGenerator):
    """LLM-based crossover and guided child prompt generator"""
    
    def forward(self,
                prompt1: str,
                wrong_examples1: str,
                gradient1: str,
                prompt2: str,
                wrong_examples2: str,
                gradient2: str,
                **kwargs) -> str:
        """
        Generate a child prompt using crossover and guidance.
        
        Args:
            prompt1: First parent prompt
            wrong_examples1: Wrong examples from first parent
            gradient1: Gradient information from first parent
            prompt2: Second parent prompt
            wrong_examples2: Wrong examples from second parent
            gradient2: Gradient information from second parent
            **kwargs: Additional arguments for LLM call
            
        Returns:
            Generated child prompt
        """
        prompt_data = self.prompt_template.format(
            prompt1=prompt1,
            prompt2=prompt2,
            wrong_examples1=wrong_examples1,
            wrong_examples2=wrong_examples2,
            gradient1=gradient1,
            gradient2=gradient2
        )
        new_prompt = self._call_llm(prompt_data, **kwargs)
        return self._parse_tagged_text(new_prompt, "<OptimizedPrompt>", "</OptimizedPrompt>")
