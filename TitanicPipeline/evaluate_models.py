import pandas as pd
import os
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime

def pr_at_k(k, result_df):
	thresh = np.percentile(result_df.score, 100-k)
			
	result_df['pred_val'] = result_df.score.apply(lambda x: 1 if x > thresh else 0)
	
	true_pos = float(len(result_df[(result_df.pred_val == 1)&(result_df.true_val==1)]))
	false_pos = len(result_df[(result_df.pred_val == 1)&(result_df.true_val==0)])
	false_neg = len(result_df[(result_df.pred_val == 0)&(result_df.true_val==1)])
	
	if true_pos+false_pos != 0:
		prec = true_pos/(true_pos+false_pos)
	else:
		prec = -1
		
	rec = true_pos/(true_pos+false_neg)
	
	return(prec ,rec)
	
def score_at_k(k, result_df, cost_mtx):
	thresh = np.percentile(result_df.score, 100-k)
	result_df['pred_val'] = result_df.score.apply(lambda x: 1 if x > thresh else 0)
	result_df['contribution'] = result_df.apply(lambda x: cost_mtx[str([int(x.pred_val), int(x.true_val)])], axis = 'columns')
	tot_score = result_df.contribution.mean()
	return tot_score
	

	
def plot_pr_curve(result_df, fn):
	p_list = []
	r_list = []
	
	l = range(1,101)
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.set_ylim(0,1)
	
	for k in l:
		pr = pr_at_k(k, result_df)
		
		p_list.append(pr[0])
		r_list.append(pr[1])
	
	ax1.plot(l, p_list, 'blue', linewidth=1.5, label = 'precision')
	ax1.plot(l, r_list, 'r--', linewidth=1.5, label = 'recall')
	ax1.set_xlabel('Percent Threshold (k)')
	
	#red_patch = mpatches.Patch(color='r--', label='recall')
	#blue_patch = mpatches.Patch(color='blue', label='precision')
	ax1.legend(loc=4)

	plt.savefig('./model_plots/'+fn+'.png')
	plt.close()

def evaluate_models():

	stream = open("evaluate_models.yml", 'r')
	docs = yaml.load_all(stream)
	for doc in docs:
		true_pos_reward = doc['true_pos_reward']
		true_neg_reward = doc['true_neg_reward']
		false_pos_penalty = doc['false_pos_penalty']
		false_neg_penalty = doc['false_neg_penalty']
		
		pic_status = doc['pics']
		k = doc['k']
		
	if os.path.isfile('./model_scores.csv'):
		model_scores_df = pd.read_csv('./model_scores.csv')
		for col in model_scores_df.columns:
			if 'Unnamed' in col:
				model_scores_df = model_scores_df.drop(col, 1)
	else:
		model_scores_df = pd.DataFrame(columns = ['model','score','recall_at_'+str(k),'precision_at_'+str(k)])
				
	filenames = []
	scores = []
	p_at_k = []
	r_at_k = []
	
	cost_mtx = {'[1, 1]': true_pos_reward, '[0, 0]': true_neg_reward, '[1, 0]': false_pos_penalty, '[0, 1]': false_neg_penalty}
	
	for fn in os.listdir('./model_output/'):
		if (os.path.isfile('./model_output/tracker.s') == False) or (os.path.isfile('./model_output/tracker.s') == True and os.path.getmtime('./model_output/'+fn) > os.path.getmtime('./model_output/tracker.s')):
			print fn
			filenames.append(fn)
			result_df_tup = pickle.load(open('./model_output/'+fn, "rb"))
			result_df = result_df_tup[0]
			top_3_feats = result_df_tup[1]
			
			
			temp_df = pd.DataFrame(columns = ['model','timestamp','score','recall_at_'+str(k),'precision_at_'+str(k), 'top_3_feats'])
			temp_df.model = [fn]
			
			pr = pr_at_k(k, result_df)
			s = score_at_k(k, result_df, cost_mtx)
			
			temp_df.score = [s]
			temp_df['precision_at_'+str(k)] = [pr[0]]
			temp_df['recall_at_'+str(k)] = [pr[1]]
			temp_df['timestamp'] = [datetime.datetime.now()]
			temp_df['top_3_feats'] = [top_3_feats]
			
			model_scores_df = model_scores_df.append(temp_df, ignore_index = True)
			model_scores_df.to_csv('model_scores.csv', index=False)
			
			if(pic_status):
				plot_pr_curve(result_df, fn[:-2])
			
	if os.path.isfile('./model_output/tracker.s') == False:
		open('./model_output/tracker.s', 'w+')
	else:
		os.utime('./model_output/tracker.s', None)
	
	
	
		