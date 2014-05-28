#Take subset of features

import data_io
import pandas as pd
from numpy import arange,array,ones,linalg
from pylab import plot,show

def main():
#	train = data_io.read_train()
#    train.fillna(0, inplace=True)

#    train_sample = train[:100000].fillna(value=0)
	data_io.get_paths()
	train_sample = data_io.read_train()
	train_sample = train_sample[train_sample['srch_id']<=600000]	
#	train_sample = pd.read_csv('train_sample.csv')
#	train_sample = train_sample[:100]
	train_sample.fillna(0, inplace=True)
	feature_names = list(train_sample.columns)
	feature_names.remove("click_bool")
	feature_names.remove("booking_bool")
	feature_names.remove("gross_bookings_usd")
	feature_names.remove("date_time")
	feature_names.remove("position")
	features = train_sample[feature_names].values
	target = train_sample["booking_bool"].values
	target2 = train_sample["click_bool"].values
	
	outputlines = []
	for i in xrange(len(features)):
		s = '{}'.format(target[i])
		s = s + ' qid:{:d} '.format(int(features[i][1]))
		for j in xrange(2,len(features[i])):
			s = s + '{}:{} '.format(j-1,features[i][j])
		outputlines.append(s)
	
	data_io.write_trainfile(outputlines)

if __name__=="__main__":
    main()

#Leave out the comp features

#trsmp = pd.read_csv('train_sample.csv')

#newtrsmp = trsmp.drop(trsmp.columns[26:], axis=1)

#criterion = trsmp['prop_location_score2'].map(lambda x: np.isnan(x))
# 
# xx = trsmp['prop_location_score1'][~criterion]
# yy = trsmp['prop_location_score2'][~criterion]
# trsmpsmall = trsmp[~criterion]
# trsmpsmall = trsmpsmall[:100]
# trsmpsmall = trsmpsmall[['prop_location_score1', 'prop_location_score2']]
# 
# scatter = vincent.Scatter(trsmpsmall, iter_idx = 'index')
