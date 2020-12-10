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

import xgboost as xgb

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.65022b1b-0ea5-4c2f-a577-49a867e3d07e"),
    inpatient_encoded_w_imputation=Input(rid="ri.foundry.main.dataset.d3578a81-014a-49a6-9887-53d296155bdd"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def data_encoded_and_outcomes(inpatient_encoded_w_imputation, outcomes):
    i = inpatient_encoded_w_imputation
    o = outcomes
    return i.join(o, on=['visit_occurrence_id'], how='inner')

@transform_pandas(
    Output(rid="ri.vector.main.execute.d699d42b-7448-42fc-85d3-ada317ac6e46"),
    data_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.65022b1b-0ea5-4c2f-a577-49a867e3d07e"),
    inpatient_encoded_w_imputation=Input(rid="ri.foundry.main.dataset.d3578a81-014a-49a6-9887-53d296155bdd"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def unnamed(data_encoded_and_outcomes, outcomes, inpatient_encoded_w_imputation):
    data_and_outcomes = data_encoded_and_outcomes
    my_data = data_and_outcomes.select(inpatient_encoded_w_imputation.columns).toPandas()
    my_data = my_data.drop(columns='visit_occurrence_id')
    my_outcomes = data_and_outcomes.select(outcomes.columns).toPandas()
    y = my_outcomes.bad_outcome
    x_train, x_test, y_train, y_test = train_test_split(my_data, y, test_size=0.3, random_state=1, stratify=y)

    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(x_train):
        xgb_model = xgb.XGBClassifier(n_jobs=1).fit(x_train[train_index], y_train[train_index])
        predictions = xgb_model.predict(x_train[test_index])
        actuals = y_train[test_index]
        print(confusion_matrix(actuals, predictions))

