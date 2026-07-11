import pandas as pd

df = pd.read_csv("results/EXP021A_2/predictions.csv")
acc = (df['pred_rank'] == 1).mean()
print(f"EXP021A_2 Accuracy: {acc:.4f}")
