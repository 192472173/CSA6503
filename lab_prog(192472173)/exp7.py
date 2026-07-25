from transformers import pipeline
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
result = classifier("I love Artificial Intelligence.")
print(result)