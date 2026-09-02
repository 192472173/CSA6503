from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
prompts = [
    "Artificial Intelligence",
    "Machine Learning",
    "Data Science"
]
for prompt in prompts:
    print("\nPrompt:", prompt)
    result = generator(prompt, max_length=50)
    print(result[0]["generated_text"])