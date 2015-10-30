import pandas as pd
import pylab as pl
import numpy as np

import re

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("./data/cs-training.csv")

# converting camelCase to snake_case.
def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

df.columns = [camel_to_snake(col) for col in df.columns]
df.columns.tolist()

#dealing with missing values
def print_null_freq(df):
    """
    for a given DataFrame, calculates how many values for 
    each variable is null and prints the resulting table to stdout
    """
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    return pd.crosstab(df_lng.variable, null_variables)
print_null_freq(df)

# assume NA for number od dependents means no dependents:
df.number_of_dependents = df.number_of_dependents.fillna(0)
# proof that the number_of_dependents no longer contains nulls
print_null_freq(df)

# for NA monthly income, use imputation
income_imputer = KNeighborsRegressor(n_neighbors=1)

# split data into training and test
is_test = np.random.uniform(0, 1, len(df)) > 0.75
train = df[is_test==False]
test = df[is_test==True]

print "len(train), len(test)"
print len(train), len(test)
print ""

# split data into 2 groups:
# data with nulls and data without
train_w_monthly_income = train[train.monthly_income.isnull()==False]
train_w_null_monthly_income = train[train.monthly_income.isnull()==True]

#print train_w_monthly_income.corr()
#print train_w_monthly_income.corr().ix[:,5]

cols = ['number_real_estate_loans_or_lines', 'number_of_open_credit_lines_and_loans']
income_imputer.fit(train_w_monthly_income[cols], train_w_monthly_income.monthly_income)

#replace missing values
new_values = income_imputer.predict(train_w_null_monthly_income[cols])
train_w_null_monthly_income['monthly_income'] = new_values
print "new_values"
print new_values
""

#combine the data back together
train = train_w_monthly_income.append(train_w_null_monthly_income)
print "len(train)"
print len(train)
print""

test['monthly_income_imputed'] = income_imputer.predict(test[cols])
#print test.head()

test['monthly_income'] = np.where(test.monthly_income.isnull(), test.monthly_income_imputed, test.monthly_income)
print "pd.value_counts(train.monthly_income.isnull())"
print pd.value_counts(train.monthly_income.isnull())
print "pd.value_counts(test.monthly_income.isnull())"
print pd.value_counts(test.monthly_income.isnull())
print ""

########################################################################
# Using RandomForest to randomly generate a "forest" of decision trees
# It calculates how much worse the model does when each variable is left out.
features = np.array(['revolving_utilization_of_unsecured_lines', 'age', 'number_of_time30-59_days_past_due_not_worse', 'debt_ratio', 'monthly_income','number_of_open_credit_lines_and_loans', 'number_of_times90_days_late', 'number_real_estate_loans_or_lines', 'number_of_time60-89_days_past_due_not_worse', 'number_of_dependents'])

clf = RandomForestClassifier()
clf.fit(train[features], train['serious_dlqin2yrs'])

# from the calculated importances, order them from most to least important
# and make a barplot so we can visualize what is/isn't important
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)

padding = np.arange(len(features)) + 0.5
pl.figure(figsize=(25,10))
pl.barh(padding, importances[sorted_idx], align='center')
pl.yticks(padding, features[sorted_idx])
pl.xlabel("Relative Importance")
pl.title("Variable Importance")
pl.savefig('feature_importances.png', bbox_inches='tight')

#From plot: best variable is revolving_utilization_of_unsecured_lines while the worst is number_real_estate_loans_or_lines with dramatic drop off after number_of_open_credit_lines_and_loans

# cap monthly income at $15,000
def cap_values(x, cap):
    if x > cap:
        return cap
    else:
        return x
    
train.monthly_income = train.monthly_income.apply(lambda x: cap_values(x, 15000))
print "train.monthly_income.describe()"
print train.monthly_income.describe()
print ""

train['income_bins'] = pd.cut(train.monthly_income, bins=15, labels=False)
print "pd.value_counts(train.income_bins)"
print pd.value_counts(train.income_bins)
print ""

print 'train[["income_bins", "serious_dlqin2yrs"]].groupby("income_bins").mean()'
print train[["income_bins", "serious_dlqin2yrs"]].groupby("income_bins").mean()
print ""

cols = ["income_bins", "serious_dlqin2yrs"]
income_means = train[cols].groupby("income_bins").mean().plot()
fig1 = income_means.get_figure()
fig1.savefig('serious_delin2yrs_vs_income.png')

cols = ['age', 'serious_dlqin2yrs']
age_means = train[cols].groupby("age").mean().plot()
fig2 = age_means.get_figure()
fig2.savefig('serious_delin2yrs_vs_age.png')

mybins = [0] + range(20, 80, 5) + [120]
train['age_bucket'] = pd.cut(train.age, bins=mybins)
print "pd.value_counts(train['age_bucket'])"
print pd.value_counts(train['age_bucket'])
print ""

# calculate the percent of customers that were delinquent for each bucket
print 'train[["age_bucket", "serious_dlqin2yrs"]].groupby("age_bucket").mean()'
print train[["age_bucket", "serious_dlqin2yrs"]].groupby("age_bucket").mean()
print ""

plt = train[["age_bucket", "serious_dlqin2yrs"]].groupby("age_bucket").mean().plot()
fig = plt.get_figure()
fig.savefig('serious_delin2yrs_vs_age_bucket.png')

labels, levels = pd.factorize(train.age_bucket)
train.age_bucket = labels

# bucket debt_ratio into 4 (nearly) equally sized groups.
bins = []

for q in [0.2, 0.4, 0.6, 0.8, 1.0]:
    bins.append(train.debt_ratio.quantile(q))

debt_ratio_binned = pd.cut(train.debt_ratio, bins=bins)
debt_ratio_binned
print "pd.value_counts(debt_ratio_binned)"
print pd.value_counts(debt_ratio_binned)
print ""

# scale columns in data frame.
train['monthly_income_scaled'] = StandardScaler().fit_transform(train.monthly_income)
print train.monthly_income_scaled.describe()
print ""
print "Mean at 0?", round(train.monthly_income_scaled.mean(), 10)==0
print ""

pl.figure()
pl.xlim([-2.0, 2.5])
pl.hist(train.monthly_income_scaled)
pl.savefig('monthly_income_scaled_hist.png')

# REDO feature importance:
features = np.array(['revolving_utilization_of_unsecured_lines', 'age', 'number_of_time30-59_days_past_due_not_worse', 'debt_ratio', 'monthly_income','number_of_open_credit_lines_and_loans', 'number_of_times90_days_late', 'number_real_estate_loans_or_lines', 'number_of_time60-89_days_past_due_not_worse', 'number_of_dependents', 'income_bins', 'age_bucket', 'monthly_income_scaled'])

clf = RandomForestClassifier()
clf.fit(train[features], train['serious_dlqin2yrs'])

importances = clf.feature_importances_
sorted_idx = np.argsort(importances)

padding = np.arange(len(features)) + 0.5
pl.figure()
pl.barh(padding, importances[sorted_idx], align='center')
pl.yticks(padding, features[sorted_idx])
pl.xlabel("Relative Importance")
pl.title("Variable Importance")
pl.savefig('feature_importances_2.png', bbox_inches='tight')

best_features = features[sorted_idx][::-1]
print "best features"
print best_features
print ""

#####################
# Fitting the model:
#####################
features = ['revolving_utilization_of_unsecured_lines', 'debt_ratio',
            'monthly_income', 'age', 'number_of_times90_days_late']

clf = KNeighborsClassifier(n_neighbors=13)
print "clf.fit(train[features], train.serious_dlqin2yrs)"
print clf.fit(train[features], train.serious_dlqin2yrs)
print ""

##########################
# generating predictions:
##########################

# classes (returns an array)
print "clf.predict(test[features])"
print clf.predict(test[features])
print ""

# probabilities (returns a numpy array)
print "clf.predict_proba(test[features])"
print clf.predict_proba(test[features])
print ""

# plot histogram of the probabilities:
probs = clf.predict_proba(test[features])
prob_true = probs[::,1]
pl.figure()
pl.hist(prob_true)
pl.savefig('probabilities.png')

########################
# Evaluating the model
########################

preds = clf.predict_proba(test[features])
print "preds: ", preds
print ""

#########################
# Classification Reports
#########################

print "confusion_matrix"
print confusion_matrix(test['serious_dlqin2yrs'], clf.predict(test[features]))
print""

print "classification report"
print classification_report(test['serious_dlqin2yrs'], clf.predict(test[features]), labels=[0, 1])
print ""

#####################################
# Evaluate Classifier
# using ROC curve
#####################################

def plot_roc(name, probs):
    fpr, tpr, thresholds = roc_curve(test['serious_dlqin2yrs'], probs)
    roc_auc = auc(fpr, tpr)
    pl.figure()
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.05])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title(name)
    pl.legend(loc="lower right")
    pl.savefig('ROC curve_'+name+'.png')

plot_roc("Perfect Classifier", test['serious_dlqin2yrs'])
plot_roc("Guessing", np.random.uniform(0, 1, len(test['serious_dlqin2yrs'])))

#[::,1] selects the 2nd column of the numpy array
plot_roc("KNN", preds[::,1])

clf = RandomForestClassifier()
print "clf.fit(train[features], train.serious_dlqin2yrs)"
print clf.fit(train[features], train.serious_dlqin2yrs)
print ""

probs = clf.predict_proba(test[features])[::,1]
plot_roc("RandomForest", probs)

# My own classifier:
features = ['revolving_utilization_of_unsecured_lines', 'debt_ratio',
            'number_of_times90_days_late', 'number_real_estate_loans_or_lines']
clf = GradientBoostingClassifier()
clf.fit(train[features], train.serious_dlqin2yrs)
probs = clf.predict_proba(test[features])[::,1]
plot_roc("My Classifier", probs)

######################################
# Convert to credit score
######################################

#odds = (1 - probs) / probs
#score = np.log(odds)*(40/np.log(2)) + 340
#pl.figure()
#pl.hist(score)
#pl.savefig('credit score.png')

def convert_prob_to_score(p):
    """
    takes a probability and converts it to a score
    Example:
        convert_prob_to_score(0.1)
        466
    """
    odds = (1 - p) / p
    scores = np.log(odds)*(40/np.log(2)) + 340
    return scores.astype(np.int)
print "probs"
print probs
print "scores"
print convert_prob_to_score(probs)
print ""
pl.figure()
pl.hist(convert_prob_to_score(probs))
pl.savefig('credit score.png')


features = ['revolving_utilization_of_unsecured_lines', 'debt_ratio',
            'monthly_income', 'age', 'number_of_times90_days_late']
clf = DecisionTreeClassifier(min_samples_leaf=1000)
print "clf.fit(train[features], train.serious_dlqin2yrs)"
print clf.fit(train[features], train.serious_dlqin2yrs)
print ""



