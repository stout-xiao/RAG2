import os
from openai import OpenAI
from config import load_config

def call_mimo_api(prompt: str, system_prompt: str = None) -> str:
    cfg = load_config()
    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.api_base
    )
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling API: {e}")
        return ""
