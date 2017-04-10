import pandas as pd
import numpy as np

def clean_data(df):
	#define which columns to drop
	cols_to_drop = ['ticket', 'body', 'name', 'home.dest', 'cabin', 'boat']
	df = df.drop(cols_to_drop, 1)
	df = df.fillna(-1)
	
	df = df[(df.T != -1).all()]
	
	df.sex = df.sex.apply(lambda x: 1 if x=='female' else 0)
	
	df.to_csv('clean_titanic.csv', index=False)
	
	
	
	

	
	