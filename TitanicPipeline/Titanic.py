import clean_data
import generate_features
import train_test_split
import generate_predictions
import evaluate_models

import pandas as pd
import os
import sys
import datetime

"""
This top level function calls all others.
"""

if len(sys.argv) > 1 and sys.argv[1] == 'clean':
	for fn in os.listdir('./'):
		if 'csv' in fn and fn != 'titanic.csv':
			print 'Removing', fn
			os.remove(fn)
		if 'pyc' in fn:
			print 'Removing', fn
			os.remove(fn)
		if '.s' in fn:
			print 'Removing', fn
			os.remove(fn)
		if '.yml' in fn:
			os.utime(fn, None)
	for fn in os.listdir('./model_output/'):
		print 'Removing', fn
		os.remove('./model_output/'+fn)
	for fn in os.listdir('./model_plots/'):
		print 'Removing', fn
		os.remove('./model_plots/'+fn)
	raise SystemExit

tasktimes = {}

proceed = False

df = pd.read_csv("titanic.csv")

#############CLEAN DATA TASK#############

if (proceed == True) or (not os.path.isfile('./clean_titanic.csv')):
	print "---------------"
	print "Cleaning Data"
	print "---------------"
	s = datetime.datetime.now()
	clean_data.clean_data(df)
	tasktimes['Data Cleaning'] = (datetime.datetime.now() - s).total_seconds()
	proceed = True
else:
	print "---------------"
	print "Cleaning Data Task Skipped"
	print "---------------"

df = pd.read_csv('clean_titanic.csv')

#########################################

#############GENERATE FEATURES TASK#############

if (proceed == True) or (not os.path.isfile('./feature_table.csv')) or (os.path.getmtime('./generate_features.yml') > os.path.getmtime('feature_table.csv')):
	print "---------------"
	print "Generating Features"
	print "---------------"
	s = datetime.datetime.now()
	generate_features.create_feature_table(df)
	tasktimes['Feature Generation'] = (datetime.datetime.now() - s).total_seconds()
	proceed = True
else:
	print "---------------"
	print "Generate Features Task Skipped"
	print "---------------"
	
df = pd.read_csv('feature_table.csv')

#########################################

#############TRAIN/TEST SPLIT TASK#############

if (proceed == True) or \
	(not os.path.isfile('train_table.csv')) or (not os.path.isfile('test_table.csv')) or \
	(os.path.getmtime('./train_test_split.yml') > min(os.path.getmtime('train_table.csv'), os.path.getmtime('test_table.csv'))):
	
	print "---------------"
	print "Generating Train/Test Sets"
	print "---------------"
	s = datetime.datetime.now()
	train_test_split.gen_train_test_tables(df)
	tasktimes['Train-Test Split'] = (datetime.datetime.now() - s).total_seconds()
	proceed = True
else:
	print "---------------"
	print "Train/Test Split Skipped"
	print "---------------"

#########################################

#############GENERATE PREDICTIONS TASK#############

if (proceed == True) or (not os.path.isfile('./generate_predictions.s')) or (os.path.getmtime('./generate_predictions.yml') > os.path.getmtime('./generate_predictions.s')):
	print "---------------"
	print "Generating Predictions"
	print "---------------"
	s = datetime.datetime.now()
	generate_predictions.run_models()
	tasktimes['Generate Predictions'] = (datetime.datetime.now() - s).total_seconds()
	proceed = True
else:
	print "---------------"
	print "Generate Predictions Task Skipped"
	print "---------------"

#########################################

#############EVALUATE MODELS TASK#############

if (proceed == True) or (not os.path.isfile('./model_scores.csv')) or (os.path.getmtime('./evaluate_models.yml') > os.path.getmtime('./model_scores.csv')):
	print "---------------"
	print "Evaluating Models"
	print "---------------"
	s = datetime.datetime.now()
	for fn in os.listdir('./model_output/'):
		if fn != 'tracker.s':
			os.utime('./model_output/'+fn, None)
	if os.path.isfile('./model_scores.csv'):
		os.remove('./model_scores.csv')
	evaluate_models.evaluate_models()
	tasktimes['Evaluate Models'] = (datetime.datetime.now() - s).total_seconds()
else:
	print "---------------"
	print "Model Evaluation Task Skipped"
	print "---------------"

#########################################

tot_time = sum(tasktimes.values())
runtime_df = pd.DataFrame()
runtime_df['Task'] = tasktimes.keys()
if tot_time != 0:
	runtime_df['Percent of Pipeline'] = [i/tot_time*100 for i in tasktimes.values()]

	print '--------------------------------------------------------------------'
	print runtime_df
	print '--------------------------------------------------------------------'
