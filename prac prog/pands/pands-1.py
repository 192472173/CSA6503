import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [20, 21, 22, 20],
    "Marks": [85, 78, 92, 88]
}

df = pd.DataFrame(data)

print(df)