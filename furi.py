import pandas as pd
pd.set_option('display.max_columns', None)
import os 
files=[f for f in os.listdir() if f.endswith('.csv')]
print("CSV files found:", files)
for file in files:
    df=pd.read_csv(file)
    print(f"\n---{file}---")
    print("Shape:",df.shape)
    print("Columns:",df.columns.tolist())
    print(df.head(2))