import os
from huggingface_hub import InferenceClient

# Set your Hugging Face token here if not using environment variables


# Initialize the client
client = InferenceClient(
    "https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud"
)

output = client.chat.completions.create(
    messages=[{"role": "user", "content": "The capital of France is"}]
)

print(output.choices[0].message.content)
