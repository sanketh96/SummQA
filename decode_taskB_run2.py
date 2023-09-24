import argparse

import pandas as pd

from TaskBSummarization import run_task_b_summarization


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', type=str, required=True)
	args = parser.parse_args()
	input_df = pd.read_csv(args.input_file)
	generated_summaries = run_task_b_summarization(input_df.dialogue.tolist())
	ids = input_df.encounter_id.tolist()
	data = {'TestID': ids, 'SystemOutput': generated_summaries}
	df = pd.DataFrame.from_dict(data)
	print('Saving predictions to csv file')
	df.to_csv('taskB_SummQA_run2.csv', index=False)