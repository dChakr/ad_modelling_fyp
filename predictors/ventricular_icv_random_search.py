# %% [markdown]
# # Regressor Model for Predicting Ventricular ICV - Random Search to find best hyperparams

# %%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

import numpy as np
import pandas as pd
import json
from scipy.io import loadmat
import sys


# %%
def compute_fc_lower_triangle(fmri, node_size):
    # Calculate the z_score (along the time axis)
    fmri_zscored = (fmri.T - fmri.mean(axis=1)) / fmri.std(axis=1)

    # Calculate the FC 
    fc = np.corrcoef(fmri_zscored.T)
    
    # Get the lower triangle
    mask_e = np.tril_indices(node_size, -1)
    lower_triangle = fc[mask_e]
    
    return lower_triangle

if __name__ == '__main__':
    # get features
    ADNI_MERGE_WITH_VENTRICULAR_VOL = sys.argv[1]
    NO_TRIALS = sys.argv[2]
    # '../data/ADNIMERGE_29Apr2024_Ventricles_ICV.csv' (use _mod version when running script as a job)

    df = pd.read_csv(ADNI_MERGE_WITH_VENTRICULAR_VOL)
    df = df[['RID', 'VISCODE', 'Ventricles_ICV', 'ABETA', 'TAU', 'DX', 'DX_bl', 'FC_DATA']]
    df = df.dropna(subset=['Ventricles_ICV', 'ABETA', 'TAU'])

    df = df.reset_index(drop=True)
    df['ABETA'] = df['ABETA'].replace('>1700', '1700')

    # %%
    dim_x = len(df)
    X = []
    NODE_SIZE = 100

    # for i, file in enumerate(df['FC_DATA'].values):
    #     arr = loadmat(file)['ROI_activity'][:NODE_SIZE, :] # get the first 100 regions
    #     fc = compute_fc_lower_triangle(arr, NODE_SIZE)
    #     X.append(fc)

    for i, file in enumerate(df['FC_DATA'].values):
        arr = loadmat(file)['ROI_activity'][:NODE_SIZE, :] # get the first 100 regions 
        fc = compute_fc_lower_triangle(arr, NODE_SIZE)
        # add abeta and tau features too
        add_features = np.asarray([float(df.loc[i, 'ABETA']), float(df.loc[i, 'TAU'])])
        features = np.concatenate((add_features, fc), axis=0)
        X.append(features)
        
    X = np.array(X)
    Y = df['Ventricles_ICV']

    # %%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    gbm = GradientBoostingRegressor()

    param_grid = {
        'n_estimators': [50, 100, 200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
        'max_depth': [2, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'subsample': np.arange(0.01, 1.0, 0.05),
        'max_features': ['sqrt', 'log2', None]
    }

    # Set Up Randomized Search
    random_search = RandomizedSearchCV(estimator=gbm, param_distributions=param_grid,
                                    n_iter=int(NO_TRIALS), cv=5, verbose=2, random_state=42,
                                    n_jobs=10, scoring='r2')

    # Fit the random search model
    random_search.fit(X_train, Y_train)

    # Print the best parameters and best score
    print("Best Parameters found: ", random_search.best_params_)
    print("Best R² score found: ", random_search.best_score_)

    # Evaluate on the test set
    y_pred = random_search.predict(X_test)
    r2 = r2_score(Y_test, y_pred)
    print(f"R² score on test set: {r2:.2f}")

    res = {
        'params': random_search.best_params_,
        'r2_score_train' : random_search.best_score_,
        'r2_score_test' : r2
    }

    with open(f'random_search_gbreg_ab_tau_{NO_TRIALS}.json', "w") as file:
        json.dump(res, file)