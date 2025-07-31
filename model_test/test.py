from openai import OpenAI

def test_cost_time(prompt):
    model_name = "ZB_Qwen3_base"
    api_key = "EMPTY"
    base_url = "http://localhost:8050/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content
print (test_cost_time("ZB_Qwen3_base:你是谁？"))
