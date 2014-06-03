
def generate_distances(tlist, sparse_dist=0):
	srch_ids = pd.unique(tlist['srch_id'])
	if (sparse_dist == 0):
		sparse_dist = sparse.lil_matrix((140821,140821), dtype=np.float64)
		
	for srch_id in srch_ids:
		query = tlist[tlist['srch_id']==srch_id][['prop_id','orig_destination_distance']]
		ix = list(query.index)
		if (math.isnan(query.at[ix[0],'orig_destination_distance'])):
			continue
		for i in ix:
			for j in ix:
				if (j>i):
					rr = int(query.loc[i,'prop_id'])
					cc = int(query.loc[j, 'prop_id'])
					dd = abs(query.loc[i, 'orig_destination_distance'] - query.loc[j, 'orig_destination_distance'])
					sparse_dist[rr,cc] = max(dd, sparse_dist[rr,cc])

	return sparse_dist					

def save_sparse_matrix(filename, x):
	with open(filename, 'wb') as outfile:
		pickle.dump(x, outfile, pickle.HIGHEST_PROTOCOL)
		
def load_sparse_matrix(filename):
	with open(filename, 'rb') as infile:
		x = pickle.load(infile)
	return x
	