import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Marks": [85, 78, 92, 88]
}

df = pd.DataFrame(data)

print("Average Marks:", df["Marks"].mean())
print("Maximum Marks:", df["Marks"].max())
print("Minimum Marks:", df["Marks"].min())