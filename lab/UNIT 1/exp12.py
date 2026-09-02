from transformers import BertTokenizer, BertModel
import torch
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
text = "Artificial Intelligence is changing the world."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state
print(embeddings.shape)