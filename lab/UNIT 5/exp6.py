from transformers import pipeline

print("LOCAL LLM QUESTION ANSWERING SYSTEM")
print("Type 'exit' to stop.")

# Load local model
chatbot = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

while True:
    question = input("\nEnter your question: ")

    if question.lower() == "exit":
        break

    prompt = (
        "You are a helpful question-answering assistant. "
        "Give clear and accurate answers.\n\n"
        "Question: " + question
    )

    response = chatbot(
        prompt,
        max_new_tokens=150
    )

    print("\nAnswer:")
    print(response[0]["generated_text"])