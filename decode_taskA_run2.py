import pandas as pd


df = pd.read_csv("taskA_SummQA_run1.csv")

processed_predictions = []
for pred in df.SystemOutput2:
  if len(pred.split(" ")) <= 5 and ("none" in pred.lower() or "noncontributory" in pred.lower() or "non-contributory" in pred.lower() or "no known" in pred.lower() or "unremarkable" in pred.lower()):
    processed_predictions.append(pred + " None / Non-contributory / Noncontributory / No known.")
  else:
    processed_predictions.append(pred)

df.SystemOutput2 = processed_predictions

df.to_csv('taskA_SummQA_run2.csv', index=False)