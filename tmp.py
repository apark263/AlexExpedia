# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/Users/alexpark/.spyder2/.temp.py
"""

import pandas as pd
import numpy as np
from numpy import arange,array,ones,linalg
from pylab import plot,show
import vincent

#Leave out the comp features

trsmp = pd.read_csv('train_sample.csv')
criterion = trsmp['prop_location_score2'].map(lambda x: np.isnan(x))
trsmpsmall = trsmp[~criterion]
trsmpsmall = trsmpsmall[:100]
trsmpsmall = trsmpsmall[['prop_location_score1', 'prop_location_score2']]

scatter = vincent.Scatter(trsmpsmall)

