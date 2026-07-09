import pandas as pd
df = pd.read_csv('results/FINAL_EVALUATION/quote_type_breakdown.csv')
print(df)
print("Sum:", df['Count'].sum())
