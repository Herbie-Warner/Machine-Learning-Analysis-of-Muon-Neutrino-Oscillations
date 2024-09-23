# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:46:19 2023

@author: herbi
"""

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from convert_to_binary_array import convert_data
from master import MC_data_full, real_data_full
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



columns_to_exclude = [feat for feat in MC_data_full.columns if feat not in real_data_full.columns]
columns_to_exclude += ['category']

n = 5000

def sample_or_all(group):
    if len(group) <= n:
        return group
    return group.sample(n)



MC_data_full = MC_data_full.groupby('category').apply(sample_or_all).reset_index(drop=True)

for col in MC_data_full.columns:
    if MC_data_full[col].dtype == 'object':
        MC_data_full[col] = pd.to_numeric(MC_data_full[col], errors='coerce')


X = MC_data_full.drop(columns_to_exclude,axis=1)
y = MC_data_full['category']

mapping = {4: 0, 5: 1, 7: 2, 10: 3, 21: 4,31: 5} 
y = y.map(mapping)



scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)




model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=10,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0.001,
    reg_lambda=1,
    objective='multi:softprob',
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)
feature_names = MC_data_full.drop(columns_to_exclude,axis=1).columns
feature_names = ['_closestNuCosmicDist', 'trk_len_v', 'trk_distance_v',
       'topological_score', 'trk_sce_end_z_v', 'trk_sce_end_y_v',
       'trk_sce_end_x_v', 'trk_score_v', 'trk_llr_pid_score_v',
       'trk_sce_start_z_v', 'trk_sce_start_y_v', 'trk_sce_start_x_v',
       'reco_nu_vtx_sce_x', 'reco_nu_vtx_sce_y', 'reco_nu_vtx_sce_z',
       'trk_energy_tot', 'trk_range_muon_mom_v', 'trk_mcs_muon_mom_v']
model.get_booster().feature_names = feature_names
ax = plot_importance(model)
fig = ax.figure
fig.set_size_inches(10, 8)
plt.show()

#feature_names = MC_data_full.drop(columns_to_exclude,axis=1).columns
#ax.set_yticklabels(feature_names)

fig.savefig('feature_importance.png',dpi=600)


# Make predictions
predictions = model.predict(X_test)
print(predictions)
print(y_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


learning_rate_range = np.arange(0.01, 1, 0.1)
test_XG = [] 
train_XG = []
for lr in learning_rate_range:
    print(lr)
    xgb_classifier = xgb.XGBClassifier(eta = lr)
    xgb_classifier.fit(X_train, y_train)
    train_XG.append(xgb_classifier.score(X_train, y_train))
    test_XG.append(xgb_classifier.score(X_test, y_test))
    
print(test_XG)
print(train_XG)
"""
fig = plt.figure(figsize=(10, 7))
plt.plot(learning_rate_range, train_XG, c='orange', label='Train')
plt.plot(learning_rate_range, test_XG, c='blue', label='Test')
plt.xlabel('Learning rate')
plt.xticks(learning_rate_range)
plt.ylabel('Accuracy score')
#plt.ylim(0.6, 1)
#plt.legend(prop={'size': 12}, loc=3)
plt.legend(loc='upper left')
plt.title('Accuracy score vs. Learning rate of XGBoost', size=14)
plt.show()

"""
learning_rate_range = np.arange(0.01, 0.5, 0.05)
fig = plt.figure(figsize=(19, 17))
idx = 1
# grid search for min_child_weight
for weight in np.arange(0, 4.5, 0.5):
    train = []
    test = []
    print(weight)
    for lr in learning_rate_range:
        xgb_classifier = xgb.XGBClassifier(eta = lr, reg_lambda=1, min_child_weight=weight)
        xgb_classifier.fit(X_train, y_train)
        train.append(xgb_classifier.score(X_train, y_train))
        test.append(xgb_classifier.score(X_test, y_test))
    fig.add_subplot(3, 3, idx)
    idx += 1
    plt.plot(learning_rate_range, train, c='orange', label='Training')
    plt.plot(learning_rate_range, test, c='m', label='Testing')
    plt.xlabel('Learning rate')
    plt.xticks(learning_rate_range)
    plt.ylabel('Accuracy score')
    #plt.ylim(0.6, 1)
    plt.legend(prop={'size': 12}, loc=3)
    title = "Min child weight:" + str(weight)
    plt.title(title, size=16)
plt.savefig('xgboost_plot.png', dpi=600)
plt.show()

