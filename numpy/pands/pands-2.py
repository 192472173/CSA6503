import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Marks": [85, 78, 92, 88]
}

df = pd.DataFrame(data)

result = df[df["Marks"] > 80]

print(result)