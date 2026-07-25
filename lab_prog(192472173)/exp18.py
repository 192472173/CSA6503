from transformers import BertTokenizer, BertModel
import torch
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
s1 = "The boy is playing football."
s2 = "A child is playing soccer."
i1 = tokenizer(s1, return_tensors="pt")
i2 = tokenizer(s2, return_tensors="pt")
e1 = model(**i1).last_hidden_state.mean(dim=1)
e2 = model(**i2).last_hidden_state.mean(dim=1)
similarity = torch.nn.functional.cosine_similarity(e1, e2)
print("Cosine Similarity:", similarity.item())