import pandas as pd
import yaml

import feature_computation

def get_feats_to_compute(feats_so_far):

	feats_to_compute = []
	
	feat_order = []
	
	stream = open("generate_features.yml", 'r')
	docs = yaml.load_all(stream)
	for doc in docs:
		if doc['incl'] == True:
			feats_to_compute.append([doc['name'], doc['deps'].replace(' ','').split(','), 0])
	
	
	while sum([i[2] for i in feats_to_compute]) != len(feats_to_compute):
		for feat in feats_to_compute:
			if set(feat[1]).issubset(feats_so_far) and feat[2]==0:
				feats_so_far.append(feat[0])
				feat_order.append(feat)
				feat[2] = 1
	
	return feat_order
	
def create_feature_table(df):
	
	feats_to_compute = get_feats_to_compute(df.columns.tolist())
	
	for feat in feats_to_compute:
		func = getattr(feature_computation, feat[0]+'_feature')
		df[feat[0]] = func(df[feat[1]])
		
	df.to_csv('feature_table.csv', index=False)