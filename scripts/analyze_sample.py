import pandas as pd

df = pd.read_csv("results/EXP010/annotated_sample.csv")

print("### Category vs Context Window Needed\n")
table1 = pd.crosstab(df['ANNOTATION: Primary Category'], df['ANNOTATION: Context Window Needed'])
print(table1.to_markdown())

print("\n### Category vs Explicit Alternative Feasible\n")
table2 = pd.crosstab(df['ANNOTATION: Primary Category'], df['ANNOTATION: Explicit Alternative Feasible?'])
print(table2.to_markdown())

print("\n### Evidence Distribution\n")
print(df['ANNOTATION: Evidence'].value_counts().to_markdown())
