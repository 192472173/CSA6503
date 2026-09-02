from transformers import pipeline
pipe = pipeline("sentiment-analysis")
output = pipe("The movie was fantastic!")
print(output)