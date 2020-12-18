import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d
import seaborn as sns
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, accuracy_score, confusion_matrix, plot_roc_curve, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline

import timeit

import xgboost as xgb

# set standard random state for repeatability
my_random_state = 42

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.65022b1b-0ea5-4c2f-a577-49a867e3d07e"),
    inpatient_encoded_w_imputation=Input(rid="ri.foundry.main.dataset.02362acb-3a3b-4fd6-ad35-677c93bd57da"),
    outcomes=Input(rid="ri.foundry.main.dataset.349f1404-e60e-4a76-9a32-13fe06198cc1")
)
def data_encoded_and_outcomes(inpatient_encoded_w_imputation, outcomes):
    i = inpatient_encoded_w_imputation
    o = outcomes
    return i.join(o, on=['visit_occurrence_id'], how='inner')

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.aac0044e-56e7-4d52-8956-712c561864f1"),
    data_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.65022b1b-0ea5-4c2f-a577-49a867e3d07e"),
    inpatient_encoded_w_imputation=Input(rid="ri.foundry.main.dataset.02362acb-3a3b-4fd6-ad35-677c93bd57da"),
    outcomes=Input(rid="ri.foundry.main.dataset.349f1404-e60e-4a76-9a32-13fe06198cc1")
)
def xbg_gs(data_encoded_and_outcomes, outcomes, inpatient_encoded_w_imputation):
    inpatient_encoded_w_imputation = inpatient_encoded_w_imputation
    start = timeit.default_timer()

    data_and_outcomes = data_encoded_and_outcomes
    my_data = data_and_outcomes.select(inpatient_encoded_w_imputation.columns).toPandas()
    my_data = my_data.drop(columns='visit_occurrence_id')
    my_outcomes = data_and_outcomes.select(outcomes.columns).toPandas()
    y = my_outcomes.bad_outcome
    x_train, x_test, y_train, y_test = train_test_split(my_data, y, test_size=0.3, random_state=my_random_state, stratify=y)

    # n_estimators - 100 default
    # learning_rate - 0.3 default
    # gamma - 0 default
    # booster – gbtree, gblinear or dart (gbtree default)
    parameters = {
        'n_estimators': range(500,1350,50),
        'learning_rate': [0.001, 0.01, 0.03, 0.07, 1.0, 2.0, 10],
        #'learning_rate': np.arange(0.001, 0.2 , 0.0025),
        'booster': ['gbtree', 'gblinear', 'dart']
    }
    # with the above options, best is:
    # {'booster': 'gbtree', 'learning_rate': 0.01, 'n_estimators': 1250}
    # took 2h 45min

    xgb_model = xgb.XGBClassifier(n_jobs=2,
                                  use_label_encoder=False,
                                  random_state=my_random_state,
                                  objective = 'binary:logistic')

    gd = GridSearchCV(estimator=xgb_model,
                      param_grid=parameters,
                      cv=5,
                      n_jobs=8,
                      verbose=3,
                      scoring='roc_auc')
    gd.fit(x_train, y_train)
    print(gd.best_params_)
    # from run before data update
    # {'booster': 'gbtree', 'learning_rate': 0.01, 'n_estimators': 1000}

    #y_pred = xgb_model.predict(x_test)
    #print(confusion_matrix(y_test, y_pred))

    #y_pred = xgb_model.predict_proba(x_test)[:, 1]
    #print('ROC_AUC_SCORE: ', roc_auc_score(y_true=y_test, y_score=y_pred))

    #stop = timeit.default_timer()
    #print('Time: ', stop - start)  

