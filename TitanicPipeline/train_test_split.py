import pandas as pd
import yaml

def gen_train_test_tables(df):
	
	stream = open("train_test_split.yml", 'r')
	docs = yaml.load_all(stream)
	for doc in docs:
		train_prop = doc['train_set_prop']
	
	df = df.sample(frac=1)
	
	thresh = int(train_prop*len(df))
	
	train_df = df[:thresh]
	test_df = df[thresh:]
	
	train_df.to_csv('train_table.csv', index=False)
	test_df.to_csv('test_table.csv', index=False)