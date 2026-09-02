from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
result = generator("Artificial Intelligence", max_length=50)
print(result[0]["generated_text"])