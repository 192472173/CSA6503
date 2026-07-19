import pandas as pd

data = {
    "Employee": ["A", "B", "C", "D", "E"],
    "Department": ["HR", "IT", "HR", "IT", "Sales"],
    "Salary": [30000, 50000, 35000, 60000, 40000]
}

df = pd.DataFrame(data)

result = df.groupby("Department")["Salary"].mean()

print(result)