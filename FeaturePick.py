#Take subset of features

import data_io
import pandas as pd
import numpy as np
from scipy import sparse
import math
import cPickle as pickle
from subprocess import call
import sys, getopt

def get_avg_vector(indices, event):
	sumcounts = np.bincount(indices.astype(int), event)
	sumtotals = np.bincount(indices.astype(int))
	return sumcounts/sumtotals

def get_cond_prob_vector(indices, event1, event2):
	sumboth = np.bincount(indices.astype(int), event1*event2)
	sumsecond = np.bincount(indices.astype(int), event2)
	condprob = sumboth
	condprob[sumsecond!=0] = condprob[sumsecond!=0]/sumsecond[sumsecond!=0]
	return condprob

def calc_prop_features(train, dev, test, recalc=0):
	if (recalc == 0):
		try:
			in_path = data_io.get_paths()["features_cache"]
			rfp = open(in_path, "r")
			px_avg    = pickle.load(rfp)
			clck_prob, book_prob, cf_prob = pickle.load(rfp)
			clck_prob_all, book_prob_all, cf_prob_all = pickle.load(rfp)
		except:
			recalc = 1	    	
	if (recalc == 1):
		# Average Price	
		prop_px_keys = ['prop_id', 'price_usd']	
		pxr = np.concatenate((train[prop_px_keys].values, dev[prop_px_keys].values, test[prop_px_keys].values))
		px_avg = get_avg_vector(pxr[:,0], pxr[:,1])

		# Click and Book Probs for just train
		prop_keys = ['prop_id', 'booking_bool', 'click_bool', 'srch_children_count']
		pbc = train[prop_keys].values
		clck_prob = get_avg_vector(pbc[:,0], pbc[:,2])
		book_prob = get_avg_vector(pbc[:,0], pbc[:,1])
		pbc[np.where(pbc[:,3]!=0)[0],3] = 1
		cf_prob = get_cond_prob_vector(pbc[:,0], pbc[:,1], pbc[:,3])

		# Click and Book Probs including dev
		pbc = np.concatenate((pbc, dev[prop_keys].values))
		clck_prob_all = get_avg_vector(pbc[:,0], pbc[:,2])
		book_prob_all = get_avg_vector(pbc[:,0], pbc[:,1])
		pbc[np.where(pbc[:,3]!=0)[0],3] = 1
		cf_prob_all = get_cond_prob_vector(pbc[:,0], pbc[:,1], pbc[:,3])


		# Account for props found in test but not in train	
		trdiff = len(px_avg) - len(clck_prob)	
		if (trdiff > 0):
			clck_prob = np.lib.pad(clck_prob, (0,trdiff), 'constant', constant_values = (0,0))
			book_prob = np.lib.pad(book_prob, (0,trdiff), 'constant', constant_values = (0,0))
			cf_prob   = np.lib.pad(cf_prob, (0,trdiff), 'constant', constant_values = (0,0))

			clck_prob_all = np.lib.pad(clck_prob_all, (0,trdiff), 'constant', constant_values = (0,0))
			book_prob_all = np.lib.pad(book_prob_all, (0,trdiff), 'constant', constant_values = (0,0))
			cf_prob_all   = np.lib.pad(cf_prob_all, (0,trdiff), 'constant', constant_values = (0,0))
				
		# Save out to cache
		out_path = data_io.get_paths()["features_cache"]
		wfp = open(out_path, "w")
		pickle.dump(px_avg, wfp)
		pickle.dump((clck_prob, book_prob, cf_prob), wfp)
		pickle.dump((clck_prob_all, book_prob_all, cf_prob_all), wfp)

	newdfkeys = ['px_avg', 'clck_prob', 'book_prob', 'cf_prob']
	newdfvals = [px_avg, clck_prob, book_prob, cf_prob]
	for setnum, dset in enumerate([train, dev, test]):
		idx = dset['prop_id'].values
		if (setnum == 2):
			newdfvals = [px_avg, clck_prob_all, book_prob_all, cf_prob_all]
		for dfkey, dfval in zip(newdfkeys, newdfvals):
			dset[dfkey] = dfval[idx]
	
	
def normalize_price_key(tlist, key, addset=None):
	px_array = tlist['price_usd'].values
	val_array = tlist[key].values
	if ((addset is None)==False):
		px_array = np.concatenate((px_array, addset['price_usd'].values))
		val_array = np.concatenate((val_array, addset[key].values))
	px_array[np.where(px_array==0)[0]] = 0.01
	px_avg = np.log10(get_avg_vector(val_array, px_array))

	if ((addset is None)==False):
		px_array = tlist['price_usd'].values
		px_array[np.where(px_array==0)[0]] = 0.01
	
	tlist['norm_px_'+key] = np.log10(px_array) - px_avg[tlist[key].values]


#normalize some numerical quantity by some category
def normalize_quantity_key(tlist, quantkey, key, addset=None):
	qx_array = tlist[quantkey].values
	val_array = tlist[key].values
	if ((addset is None)==False):
		qx_array = np.concatenate((qx_array, addset[quantkey].values))
		val_array = np.concatenate((val_array, addset[key].values))
	qx_avg = get_avg_vector(val_array, qx_array)

	if ((addset is None)==False):
		qx_array = tlist[quantkey].values
	
	tlist['norm_' + quantkey + '_' +key] = qx_array/qx_avg[tlist[key].values]


def replace_history_features(tlist):
	tlist['visitor_hist_starrating'] = abs(tlist['visitor_hist_starrating']-tlist['prop_starrating'])	
	tlist['visitor_hist_adr_usd'] = abs(tlist['visitor_hist_adr_usd']-tlist['price_usd'])

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


def augment_data(tlist, addset=None):
	replace_history_features(tlist)
	for setname in ['srch_id', 'prop_id', 'srch_destination_id']:
		normalize_price_key(tlist, setname, addset)
	
	for setname in ['srch_id', 'srch_destination_id']:
		for quantkey in ['prop_location_score1', 'prop_review_score', 'prop_starrating', 'prop_log_historical_price']:
			normalize_quantity_key(tlist, quantkey, setname, addset)

	for setname in ['prop_id', 'srch_destination_id']:
		for quantkey in ['srch_booking_window', 'srch_query_affinity_score']:
			normalize_quantity_key(tlist, quantkey, setname, addset)

	for setname in ['srch_id', 'prop_id', 'srch_destination_id']:
		for quantkey in ['orig_destination_distance']:
			normalize_quantity_key(tlist, quantkey, setname, addset)

				
def LR_score_to_submission():
	test = data_io.read_test()
	score_file = data_io.get_paths()["score_path"]
	submission_path = data_io.get_paths()["submission_path"]
	scores = pd.read_csv(score_file, delimiter='\t', header = None)
	test['rscores'] = scores[2]
	result = test.sort_index(by=['srch_id','rscores'], ascending=[True, False])
	output = result[['srch_id','prop_id']]
	output.columns = ['SearchId', 'PropertyId']	
	output.to_csv(submission_path, index = None)


def dump_train_data():
	train = data_io.read_train()
	test = data_io.read_test()
	
	dev = train[train['srch_id'] % 10 == 1].copy()
	train_only = train[train['srch_id'] % 10 != 1].copy()
	
	# get per Property features over provided data
	calc_prop_features(train_only, dev, test)
	
	# Supplement data frame with extra features
	
	augment_data(train_only)
	for dataset in [dev, test]:
		augment_data(dataset, train_only)	

	feature_names = list(train_only.columns)
	feature_names.remove("click_bool")
	feature_names.remove("booking_bool")
	feature_names.remove("gross_bookings_usd")
	feature_names.remove("date_time")
	feature_names.remove("position")
	feature_names = filter(lambda a: a.startswith('comp')==False, feature_names)
	
	train_only.fillna(0, inplace=True)
	dev.fillna(0, inplace=True)
	test.fillna(0, inplace=True)
	dev_target = np.maximum(dev['booking_bool'].values*5, dev['click_bool'].values)
	
	#downsampled training examples
	train_index = train_only[(train_only['booking_bool'] != 0) | (train_only['click_bool'] != 0) | (train_only.index % 3 == 1)].index
	train_target = np.maximum(train_only.loc[train_index]['booking_bool'].values*5, train_only.loc[train_index]['click_bool'].values)
	print "Dev/Train: ", len(dev), len(train_index)
	
	save_data_to_ranklib(dev[feature_names].values, dev_target, 'RLdev')
	save_data_to_ranklib(train_only.loc[train_index][feature_names].values, train_target, 'RLtrain')	
	save_data_to_ranklib(test[feature_names].values, np.zeros(len(test)), 'RLtest')

def train_model():
	jv_exec = data_io.get_paths()["java_path"]
	rank_lib = data_io.get_paths()["ranklib_path"]
	model_file = data_io.get_paths()["model_path"]
	train_file = data_io.get_paths()["RLtrain_path"]
	dev_file = data_io.get_paths()["RLdev_path"]
	
	cmdstring = jv_exec
	cmdstring += " -jar " + rank_lib

	argstring  = " -train " + train_file
	argstring += " -validate " + dev_file
	argstring += " -save " + model_file
	argstring += " -ranker 6 -metric2t NDCG@38 "

	print "Begin Training"
	print "Running " + cmdstring + argstring

	status = call(cmdstring + argstring, shell=True)
	

def test_model():
	jv_exec = data_io.get_paths()["java_path"]
	rank_lib = data_io.get_paths()["ranklib_path"]
	model_file = data_io.get_paths()["model_path"]
	test_file = data_io.get_paths()["RLtest_path"]
	score_file = data_io.get_paths()["score_path"]

	cmdstring = jv_exec
	cmdstring += " -jar " + rank_lib
	
	argstring  = " -load " + model_file
	argstring += " -rank " + test_file
	argstring += " -score " + score_file

	print "Begin Testing"
	print "Running " + cmdstring + argstring

	status = call(cmdstring + argstring, shell=True)
			
def main(argv):
	mode = ''
	usage_string = 'FeaturePick.py -m <train|test|data>'
	
	try:
		opts, args = getopt.getopt(argv,"m:",["mode="])
	except getopt.GetoptError:
		print usage_string
		sys.exit(2)	
	for opt, arg in opts:
		if opt in ("-m", "--mode"):
			mode = arg
			
	if (mode == 'train'):
		train_model()
	elif (mode == 'test'):
		test_model()
		
		#LR_score_to_submission()
	elif (mode == 'data'):
		print "Generating Training data"
		dump_train_data()
	else:
		print usage_string
		
	
if __name__=="__main__":
    main(sys.argv[1:])

#TODO Normalized price
#	
#Leave out the comp features

#trsmp = pd.read_csv('train_sample.csv')

#newtrsmp = trsmp.drop(trsmp.columns[26:], axis=1)
