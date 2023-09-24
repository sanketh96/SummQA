import argparse

import pandas as pd

from TaskAClassification import run_task_A_classification
from TaskASummarization import run_task_A_summarization

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', type=str, required=True)
	args = parser.parse_args()
	input_df = pd.read_csv(args.input_file)
	predicted_classes = run_task_A_classification(input_df.dialogue.tolist())
	generated_summaries = run_task_A_summarization(input_df.dialogue.tolist())
	ids = input_df.ID.tolist()
	data = {'TestID': ids, 'SystemOutput1': predicted_classes, 'SystemOutput2': generated_summaries}
	df = pd.DataFrame.from_dict(data)
	print('Saving predictions to csv file')
	df.to_csv('taskA_SummQA_run1.csv', index=False)
