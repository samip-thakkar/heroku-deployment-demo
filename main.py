# -*- coding: utf-8 -*-
"""

@author: Samip
"""

import pandas as pd
data = pd.read_csv('insurance.csv')

from pycaret.regression import *

#Experiment 1
s = setup(data, target = 'charges', session_id = 123)

lr = create_model('lr')

plot_model(lr)

#Experiment 2
s2 = setup(data, target = 'charges', session_id = 123, normalize = True, polynomial_features = True, trigonometry_features = True, feature_interaction = True,            bin_numeric_features= ['age', 'bmi'])
lr = create_model('lr')

plot_model(lr)

save_model(lr, 'deployment_05022020')
deployment_05022020 = load_model('deployment_05022020')


import requests
url = 'https://insurance-price-prediction.herokuapp.com/'
pred = requests.post(url, json = {'age':55, 'sex':'male', 'bmi':59, 'children':1, 'smoker':'male', 'region':'northwest'})
print(pred.json())

