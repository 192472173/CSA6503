from transformers import pipeline
qa = pipeline("question-answering")
context = """
Python is a programming language created by Guido van Rossum.
"""
result = qa(
    question="Who created Python?",
    context=context
)
print(result)