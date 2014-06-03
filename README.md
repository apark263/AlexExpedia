AlexExpedia
===========
Runs feature extraction, training, testing, and submission formatting for Expedia challenge.

Requires:
java, RankLib, Kaggle Expedia Data

1) Set all path information in SETTINGS.json

2) Extract features and generate data in RankLib format
    python FeaturePick.py data

3) Train LambdaMart model
    python FeaturePick.py train
    
4) Test LambdaMart against test data and create submission csv file
    python FeaturePick.py test

