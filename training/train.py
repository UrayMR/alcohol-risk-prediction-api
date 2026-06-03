import pandas as pd

df = pd.read_csv("data/raw/dataset.csv")

# Drop rows with missing values
df = df.dropna()

