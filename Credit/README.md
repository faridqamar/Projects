Predict if someone will experience financial distress in the next two years

Kaggle competition 2011:
Improve on the state of the art in credit scoring by predicting the probability that somebody will experience financial distress in the next two years:
 
Banks play a crucial role in market economies. They decide who can get finance and on what terms and can make or break investment decisions. For markets and society to function, individuals and companies need access to credit. 

Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. This competition requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.

The goal of this competition is to build a model that borrowers can use to help make the best financial decisions.

Historical data are provided on 250,000 borrowers.

The code:
- loads the data with missing monthly_income
- splits the data into a trainig set for building the model and a test set for evaluating its performance
- Uses KNearestNeighbors to fill in missing values for monthly_income
- ranks feature importance using RandomForest
- trains a credit classifier
- evaluates the results by both plotting data and generating reports 
- converts the model into a credit score


