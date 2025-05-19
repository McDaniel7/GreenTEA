# GreenTEA/utils/llm_utils.py

import boto3
import json
import time
import logging
from typing import Union, List, Dict, Any
from langchain_core.prompts import PromptTemplate

class LLMPromptRephraser:
    """LLM-based prompt rephrasing utility"""
    
    def __init__(self, rephraser_template_path: str, model_name: str):
        """
        Initialize prompt rephraser.
        
        Args:
            rephraser_template_path: Path to rephraser template file
            model_name: Name of the LLM model to use
        """
        self.model_name = model_name
        self.client = boto3.client('bedrock-runtime')
        
        with open(rephraser_template_path, 'r') as f:
            ptxt = f.read()
            self.prompt_template = PromptTemplate.from_template(ptxt)

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
        """Call LLM with retry logic"""
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

    def forward(self, sentence: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        Rephrase input prompt(s).
        
        Args:
            sentence: Input prompt or list of prompts
            **kwargs: Additional arguments for LLM call
            
        Returns:
            Rephrased prompt(s)
        """
        if isinstance(sentence, list):
            results = [self.forward(sen, **kwargs) for sen in sentence]
        else:
            prompt_data = self.prompt_template.format(prompt=sentence)
            new_prompt = self._call_llm(prompt_data, **kwargs)
            try:
                results = self._parse_tagged_text(new_prompt, "<NewPrompt>", "</NewPrompt>")
            except Exception as e:
                results = new_prompt
        
        return results
