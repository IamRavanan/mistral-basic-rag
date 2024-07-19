from mistralai.client import MistralClient
import os

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"

client = MistralClient(api_key=api_key)

embeddings_response = client.embeddings(
    model=model,
    input=["Embed this sentence.", "As well as this one."]
)

print(embeddings_response)