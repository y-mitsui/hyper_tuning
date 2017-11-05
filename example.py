from xgboost import XGBRegressor
from tree_cut import ThompsonParameters
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(123451)
xbg_tuning_parameters = {
    'max_depth' : { "id" : "max_depth",
                    "type": "gp",
                    "conditions" : ((2, 20), )
                  },
    'subsample' : { "id":"subsample",
                    "type" : "gp",
                    "conditions" : ((0.5, 1), )
                  },
    'reg_lambda' : { "id" : "reg_lambda",
                    "conditions" : np.linspace(0.0, 1.5, 15)},
    'reg_alpha' : { "id" : "reg_alpha",
                    "conditions" : np.linspace(0.0, 1.0, 10)},
    'learning_rate' : { "id" : "learning_rate",
                    "conditions" : np.linspace(0.01, 1, 10)},
    'gamma' : { "id" : "gamma",
                    "conditions" : np.linspace(0, 2, 20)},
    'min_child_weight' : { "id" : "min_child_weight",
                    "conditions" : np.linspace(0, 10, 10)},
    'colsample_bytree' : { "id" : "colsample_bytree",
                    "conditions" : np.linspace(0.3, 1, 5)},
}

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target)

scores = []
reg = XGBRegressor(silent=1, nthread=-1)
thompson_parameters = ThompsonParameters(xbg_tuning_parameters, 80, 20)
while thompson_parameters.hasNext():
    params_obj = thompson_parameters.getParameters()
    cur_parameters = dict(params_obj["parameters"])
    cur_parameters["max_depth"] = int(round(cur_parameters["max_depth"][0]))
    cur_parameters["subsample"] = cur_parameters["subsample"][0]
    print(cur_parameters["max_depth"], cur_parameters["subsample"])
    reg.set_params(**cur_parameters)
    reg.fit(X_train, y_train)
    
    score = reg.score(X_test, y_test)
    thompson_parameters.setScore(params_obj, score)
    
    scores.append(score)
    
print(zip(thompson_parameters.bayes_opt["max_depth"].X, thompson_parameters.bayes_opt["max_depth"].Y))
plt.plot(scores)
plt.show()
