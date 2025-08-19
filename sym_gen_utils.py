import os
from openai import OpenAI

openai_api_key = os.getenv('OPENAI_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_KEY')
client = OpenAI(api_key=openai_api_key)
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url='https://api.deepseek.com')


def get_response(prompt, system_message, args, model_name='gpt-4.1-nano'):
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
    elif model_name in ['deepseek-chat']:
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
    elif model_name in ['o3', 'o3-mini', 'o4-mini']:
        response = client.responses.create(
            model=model_name,
            input=prompt,
            instructions=system_message,
            reasoning={"effort": args.reasoning_effort, "summary": None}
        )
        output = response.output_text
        return output
    else: #openai models
        response = client.responses.create(
            model=model_name,
            input=prompt,
            instructions=system_message,
            temperature=args.temperature,
            top_p=args.top_p
        )

        #return response.output[0].content[0].text
        return response.output_text