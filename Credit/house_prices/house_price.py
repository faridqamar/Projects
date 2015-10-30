import pandas as pd
import numpy as np
import pylab as pl

import re

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

# converting camelCase to snake_case.
def camel_to_snake(column_name):
	s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
	return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

boston = load_boston()

#    Create a dataframe with the Boston data
bostonframe = pd.DataFrame(boston.data)
#    snake_case/lower_case the columns
bostonframe.columns = [camel_to_snake(col) for col in boston.feature_names]
# add in prices
bostonframe['price'] = boston.target
#print len(bostonframe)==506
#print bostonframe.head()


#    define features
features = ['age', 'lstat', 'tax']
#    create a LinearRegression
lm = LinearRegression()
#    fit the model
lm.fit(bostonframe[features], bostonframe.price)

# add actual vs. predicted points
pl.scatter(bostonframe.price, lm.predict(bostonframe[features]))
# add the line of perfect fit
straight_line = np.arange(0, 60)
pl.plot(straight_line, straight_line)
pl.title("Fitted Values")
pl.savefig('fitted_house_prices.png')




