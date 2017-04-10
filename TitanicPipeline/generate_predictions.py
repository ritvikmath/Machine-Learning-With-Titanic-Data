import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import pickle
from sklearn.grid_search import ParameterGrid
import os

def run_models():
	stream = open("generate_predictions.yml", 'r')
	docs = yaml.load_all(stream)
	
	for doc in docs:		
		train_df = pd.read_csv(doc['train_table_name'])
		test_df = pd.read_csv(doc['test_table_name'])
		models_to_run = doc['models_to_run'].replace(' ','').split(',')
		prediction_var = doc['prediction_var']
		feats_to_use = doc['feats_to_use'].replace(' ','').split(',')
	
	X_train = train_df[feats_to_use]
	X_test = test_df[feats_to_use]
	
	y_train = train_df[prediction_var]
	y_test = test_df[prediction_var]
	
	clfs, grid = define_clfs_params()
	
	for n in range(1, 2):
		for index,clf in enumerate([clfs[x] for x in models_to_run]):
			parameter_values = grid[models_to_run[index]]
			for p in ParameterGrid(parameter_values):
				try:
					filename = models_to_run[index]+'-'+str(p).replace(' ','').strip('{}').replace('\'','').replace(',','-').replace(':','_')+'-'+'+'.join(feats_to_use)
					if os.path.isfile("./model_output/"+filename+".p"):
						continue
					print clf
					clf.set_params(**p)
					y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
					try:
						zipped_imps = sorted(zip(X_train.columns,clf.feature_importances_), key = lambda x:x[1])
						top_3_feats = [i[0] for i in zipped_imps[:3]]
					except AttributeError:
						top_3_feats = ['NA']
					print "---------------"
					result = pd.DataFrame()
					result['true_val'] = y_test
					result['score'] = y_pred_probs
					pickle.dump( [result, top_3_feats], open("./model_output/"+filename+".p", "wb" ))
				except IndexError, e:
					print 'Error:',e
					continue
					
	open('generate_predictions.s', 'w+')
	
def define_clfs_params():

	clfs = {
		'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
		'LR': LogisticRegression(penalty='l1', C=1e5),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
		'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
		'KNN': KNeighborsClassifier(n_neighbors=3),
		'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
		
	}

	grid = { 
	'RF':{'n_estimators': [10,100,1000], 'max_depth': [2,5], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
	'LR': { 'penalty': ['l1','l2'], 'C': [.1,1,10]},
	'GB': {'n_estimators': [10,100], 'learning_rate' : [0.01,0.05],'subsample' : [0.1,0.5], 'max_depth': [10,50]},
	'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [10,20,50, 100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
	'KNN' :{'n_neighbors': [50,100,200],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
	'AB': { 'algorithm': ['SAMME','SAMME.R'], 'n_estimators': [1,10,100,1000,10000]}
	}
	
	return clfs, grid
	
if __name__ == "__main__":
    run_models()