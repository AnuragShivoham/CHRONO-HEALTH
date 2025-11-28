import pandas as pd

path = r"C:\Users\ANURAG\Dropbox\MediBELL_DATABASE\Data\processed\label_mapping_25k.csv"

print("Reading:", path)

try:
    df = pd.read_csv(path)
    print("\nColumns:", df.columns.tolist())
    print("\nRows:")
    print(df)
except Exception as e:
    print("\n❌ ERROR:", e)
