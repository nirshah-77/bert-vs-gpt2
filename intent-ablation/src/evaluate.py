import pandas as pd
from config import RESULTS_CSV

df = pd.read_csv(RESULTS_CSV)
frozen = df[df["strategy"] == "frozen"]
summary = frozen.groupby("model")["test_acc"].agg(["mean", "std"])
print(summary)