from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sentence = "Natural Language Processing is interesting."
print(tokenizer.tokenize(sentence))