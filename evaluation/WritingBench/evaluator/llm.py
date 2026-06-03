import time
from typing import Callable
from openai import OpenAI


class ClaudeAgent(object):
    def __init__(self,
                 system_prompt: str = None,
                 api_key: str = None,
                 url: str = None,
                 model: str = None):
        self.system_prompt = system_prompt
        self.api_key = api_key or '' # Your API KEY
        self.base_url = url or '' # Your URL path (base_url for OpenAI client)
        self.model = model or '' # Model name
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def call_claude(self,
             messages: list,
             top_p: float = 0.95,
             temperature: float = 1.0,
             max_length: int = 2048):
        attempt = 0
        max_attempts = 5
        wait_time = 1

        while attempt < max_attempts:
            try:
                # Debug: Print request details (without sensitive info)
                # print(f"DEBUG: Attempt {attempt+1}/{max_attempts}")
                # print(f"DEBUG: Base URL: {self.base_url}")
                # print(f"DEBUG: Model: {self.model}")
                # print(f"DEBUG: API Key present: {bool(self.api_key)}")
                # print(f"DEBUG: Messages count: {len(messages)}")
                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_length,
                    top_p=top_p,
                    temperature=temperature,
                )
                
                return completion.choices[0].message.content
            
            except Exception as e:
                print(f"Attempt {attempt+1}: Request failed: {type(e).__name__}: {e}, retrying...")
                time.sleep(wait_time)
                attempt += 1

        raise Exception("Max attempts exceeded. Failed to get a successful response.")
    
    def basic_success_check(self, response):
        if not response:
            print(response)
            return False
        else:
            return True
    
    def run(self,
            prompt: str,
            top_p: float = 0.95,
            temperature: float = 1.0,
            max_length: int = 2048,
            max_try: int = 5,
            success_check_fn: Callable = None):
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user","content": prompt}
        ]
        success = False
        try_times = 0

        while try_times < max_try:
            response = self.call_claude(
                messages=messages,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
            )

            if success_check_fn is None:
                success_check_fn = lambda x: True
            
            if success_check_fn(response):
                success = True
                break
            else:
                try_times += 1
        
        return response, success
