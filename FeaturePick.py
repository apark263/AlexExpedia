#Take subset of features

import data_io
import pandas as pd
import numpy as np
from scipy import sparse
import math
import cPickle as pickle
from subprocess import call

def calc_prop_features(train, test, recalc=0):
	if (recalc == 0):
		try:
			in_path = data_io.get_paths()["features_cache"]
			rfp = open(in_path, "r")
			px_avg    = pickle.load(rfp)
			clck_prob = pickle.load(rfp)
			book_prob = pickle.load(rfp)
			clck_prob_all = pickle.load(rfp)
			book_prob_all = pickle.load(rfp)
		except:
			recalc = 1	    	
	if (recalc == 1):
		# Average Price
		tlist = pd.concat((train[['prop_id','price_usd']], test[['prop_id','price_usd']]))
		px_sum = np.bincount(tlist['prop_id'].values, tlist['price_usd'].values)
		px_cnt = np.bincount(tlist['prop_id'].values)
		px_avg = px_sum/px_cnt

		# Click and Book Probs
		train_nodev = train[(train['srch_id'] % 10 != 1)][['prop_id', 'booking_bool', 'click_bool']]
		book_sum = np.bincount(train_nodev['prop_id'].values, train_nodev['booking_bool'].values)	
		clck_sum = np.bincount(train_nodev['prop_id'].values, train_nodev['click_bool'].values)
		clck_cnt = np.bincount(train_nodev['prop_id'].values)
		clck_prob = clck_sum/clck_cnt
		book_prob = book_sum/clck_cnt

		book_sum = np.bincount(train['prop_id'].values, train['booking_bool'].values)	
		clck_sum = np.bincount(train['prop_id'].values, train['click_bool'].values)
		clck_cnt = np.bincount(train['prop_id'].values)
		clck_prob_all = clck_sum/clck_cnt
		book_prob_all = book_sum/clck_cnt

		# Account for props found in test but not in train	
		trdiff = len(px_avg) - len(clck_prob)	
		if (trdiff > 0):
			clck_prob = np.lib.pad(clck_prob, (0,trdiff), 'constant', constant_values = (0,0))
			book_prob = np.lib.pad(book_prob, (0,trdiff), 'constant', constant_values = (0,0))
			clck_prob_all = np.lib.pad(clck_prob_all, (0,trdiff), 'constant', constant_values = (0,0))
			book_prob_all = np.lib.pad(book_prob_all, (0,trdiff), 'constant', constant_values = (0,0))
			
	
		# Save out to cache
		out_path = data_io.get_paths()["features_cache"]
		wfp = open(out_path, "w")
		pickle.dump(px_avg, wfp)
		pickle.dump(clck_prob, wfp)
		pickle.dump(book_prob, wfp)
		pickle.dump(clck_prob_all, wfp)
		pickle.dump(book_prob_all, wfp)

	return px_avg, clck_prob, book_prob, clck_prob_all, book_prob_all

def normalize_price_key(tlist, key, addset=None):
	px_array = tlist['price_usd'].values
	val_array = tlist[key].values
	if ((addset is None)==False):
		px_array = np.concatenate((px_array, addset['price_usd'].values))
		val_array = np.concatenate((val_array, addset[key].values))
	px_array[np.where(px_array==0)[0]] = 0.01
	px_sum = np.bincount(val_array, px_array)
	px_cnt = np.bincount(val_array)
	px_avg = np.log10(px_sum/px_cnt)

	if ((addset is None)==False):
		px_array = tlist['price_usd'].values
		px_array[np.where(px_array==0)[0]] = 0.01
	
	tlist['norm_px_'+key] = np.log10(px_array) - px_avg[tlist[key].values]


def replace_history_features(tlist):
	tlist['visitor_hist_starrating'] = abs(tlist['visitor_hist_starrating']-tlist['prop_starrating'])	
	tlist['visitor_hist_adr_usd'] = abs(tlist['visitor_hist_adr_usd']-tlist['price_usd'])
	return

def save_data_to_ranklib(features, target, RLset):
	outputlines = []
	nf = len(features[0][1:])
	# Feature Indexes
	f_idx = np.char.mod('%d:', range(1,nf+1))

	for i in xrange(len(features)):
		s = '{} qid:{:d} '.format(target[i], int(features[i][0]))
		# Interleave features and indices and output as string
		f_vals = np.char.mod('%g ', features[i][1:])
		s = s + "".join(np.reshape(np.transpose([f_idx,f_vals]), nf*2)) + "\n"
		outputlines.append(s)
	
	data_io.write_RLfile(outputlines, RLset)

def augment_data(tlist, px_avg, clck_prob, book_prob, addset=None):
	tlist['px_avg'] = px_avg[tlist['prop_id'].values]
	tlist['clck_prob'] = clck_prob[tlist['prop_id'].values]
	tlist['book_prob'] = book_prob[tlist['prop_id'].values]

	replace_history_features(tlist)
	normalize_price_key(tlist, 'srch_id', addset)
	normalize_price_key(tlist, 'prop_id', addset)
	normalize_price_key(tlist, 'srch_destination_id', addset)
				
def LR_score_to_submission():
	test = data_io.read_test()
	score_file = data_io.get_paths()["score_path"]
	scores = pd.read_csv(score_file, delimiter='\t', header = None)
	test['rscores'] = scores[2]
	result = test.sort_index(by=['srch_id','rscores'], ascending=[True, False])
	output = result[['srch_id','prop_id']]
	output.columns = ['SearchId', 'PropertyId']	
	output.to_csv('tmp.csv', index = None)


def dump_train_data():
	train = data_io.read_train()
	test = data_io.read_test()

	# get per Property features over provided data
	px_avg, clck_prob, book_prob, clck_prob_all, book_prob_all = calc_prop_features(train, test)
	
	# Supplement data frame with extra features
	augment_data(train, px_avg, clck_prob, book_prob)
	augment_data(test,  px_avg, clck_prob_all, book_prob_all, train)

	feature_names = list(train.columns)
	feature_names.remove("click_bool")
	feature_names.remove("booking_bool")
	feature_names.remove("gross_bookings_usd")
	feature_names.remove("date_time")
	feature_names.remove("position")
	feature_names = filter(lambda a: a.startswith('comp')==False, feature_names)
	
	train.fillna(0, inplace=True)
	test.fillna(0, inplace=True)
	train_target = np.maximum(train['booking_bool'].values*5, train['click_bool'].values)
	
	dev_index = train[train['srch_id'] % 10 == 1].index
	#downsampled training examples
	train_index = train[(train['srch_id'] %10 != 1) & ((train['booking_bool'] != 0) | (train['click_bool'] != 0) | (train.index % 3 == 1))].index
	print "Dev/Train: ", len(dev_index), len(train_index)
	
	save_data_to_ranklib(train.iloc[dev_index][feature_names].values, train_target[dev_index], 'RLdev')
	save_data_to_ranklib(train.iloc[train_index][feature_names].values, train_target[train_index], 'RLtrain')	
	save_data_to_ranklib(test[feature_names].values, np.zeros(len(test)), 'RLtest')

def train_model():
	jv_exec = data_io.get_paths()["features_cache"]
	rank_lib = data_io.get_paths()["ranklib_path"]
	model_file = data_io.get_paths()["model_path"]
	train_file = data_io.get_paths()["RLtrain_path"]
	dev_file = data_io.get_paths()["RLdev_path"]
	
	cmdstring = jv_exec
	argstring += " -jar " + rank_lib

	argstring  = " -train " + train_file
	argstring += " -validate " + dev_file
	argstring += " -save " + model_file
	argstring += " -ranker 6 -metric2t NDCG@38 "

	print "Begin Training"
	print "Running " + cmdstring + argstring

	status = call(cmdstring + argstring)
	

def test_model():
	jv_exec = data_io.get_paths()["features_cache"]
	rank_lib = data_io.get_paths()["ranklib_path"]
	model_file = data_io.get_paths()["model_path"]
	test_file = data_io.get_paths()["RLtest_path"]

	cmdstring = jv_exec
	argstring += " -jar " + rank_lib
	
	argstring += " -load " + model_file
	argstring += " -rank " + test_file
	argstring += " -score " + score_file

	print "Begin Testing"
	print "Running " + cmdstring + argstring

	status = call(cmdstring + argstring)
		
	
def main():
	
	LR_score_to_submission()
		
	
if __name__=="__main__":
    main()

#TODO Normalized price
#	
#Leave out the comp features

#trsmp = pd.read_csv('train_sample.csv')

#newtrsmp = trsmp.drop(trsmp.columns[26:], axis=1)
