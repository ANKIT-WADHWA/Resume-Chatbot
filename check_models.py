from groq import Groq

client = Groq(api_key="YOUR_GROQ_API_KEY")

models = client.models.list()
print(models)
