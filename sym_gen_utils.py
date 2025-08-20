import os
from openai import OpenAI

openai_api_key = os.getenv('OPENAI_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_KEY')
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
client = OpenAI(api_key=openai_api_key)
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url='https://api.deepseek.com')
openrouter_client = OpenAI(api_key=openrouter_api_key, base_url='https://openrouter.ai/api/v1')


def get_response(prompt, system_message, args, model_name='gpt-4.1-nano', platform=None):
    # Determine platform from model_name if not explicitly provided
    if platform is None:
        if '/' in model_name:
            platform = 'openrouter'
        elif model_name in ['deepseek-reasoner', 'deepseek-chat']:
            platform = 'deepseek'
        elif model_name in ['o3', 'o3-mini', 'o4-mini']:
            platform = 'anthropic'
        else:
            platform = 'openai'
    
    # Handle different platforms
    if platform == 'openrouter':
        actual_model = model_name.replace('openrouter/', '') if model_name.startswith('openrouter/') else model_name
        
        # Check if this is a reasoning model (like Qwen thinking model)
        reasoning_models = ['qwen/qwen3-235b-a22b-thinking-2507']
        is_reasoning_model = any(reasoning_model in actual_model for reasoning_model in reasoning_models)
        
        # Prepare API parameters
        api_params = {
            "model": actual_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": getattr(args, 'temperature', 0.7),
            "top_p": getattr(args, 'top_p', 1.0)
        }
        
        # Add reasoning parameters for reasoning models
        if is_reasoning_model:
            reasoning_effort = getattr(args, 'reasoning_effort', 'medium')
            # Use extra_body parameter to pass reasoning config (OpenRouter specific)
            api_params["extra_body"] = {
                "reasoning": {
                    "effort": reasoning_effort,  # Can be "high", "medium", or "low"
                    "exclude": False  # Include reasoning tokens in response
                }
            }
        
        response = openrouter_client.chat.completions.create(**api_params)
        output = response.choices[0].message.content
        
        # Extract reasoning content if available
        reasoning_content = None
        if is_reasoning_model and hasattr(response.choices[0].message, 'reasoning'):
            reasoning_content = response.choices[0].message.reasoning
        
        # Return both output and reasoning content for reasoning models
        if is_reasoning_model:
            return output, reasoning_content
        else:
            return output
            
    elif platform == 'deepseek':
        if model_name == 'deepseek-reasoner':
            response = deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            reasoning_content = response.choices[0].message.reasoning_content
            output = response.choices[0].message.content
            return output, reasoning_content
        else:  # deepseek-chat
            response = deepseek_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message}, 
                    {"role": "user", "content": prompt}
                ],
                temperature=args.temperature,
            )
            output = response.choices[0].message.content
            return output
    elif platform == 'anthropic':
        response = client.responses.create(
            model=model_name,
            input=prompt,
            instructions=system_message,
            reasoning={"effort": args.reasoning_effort, "summary": None}
        )
        output = response.output_text
        return output
    else:  # openai platform
        response = client.responses.create(
            model=model_name,
            input=prompt,
            instructions=system_message,
            temperature=args.temperature,
            top_p=args.top_p
        )
        return response.output_text