from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
prompt = "The future of Artificial Intelligence"
result = generator(prompt, max_length=80)
print(result[0]["generated_text"])