import pandas as pd
import numpy as np 
import json
from scipy.io import loadmat
import sys
from sklearn.model_selection import train_test_split

from whobpyt.datatypes import par, Recording
from whobpyt.models.RWWABT import RNNRWWABT, ParamsRWWABT
from whobpyt.optimization.custom_cost_RWW import CostsRWW
from whobpyt.run import Model_fitting
import optuna
import torch

def get_data(DATA_PATH, pgroup = 'CN'):
    patients = pd.read_csv(DATA_PATH)
    patients = patients[patients['DX_bl'] == pgroup]

    fcs = []

    for f in patients['FC_DATA']:
        # when you're not running this with a script, remember to add ../FMRI_ADNI_DATA/ before the filenames
        fc = loadmat(f)
        fc = fc['ROI_activity'][:100, :]
        if fc.shape[1] == 197:
            fc = fc.T
            fcs.append(fc)
        
    # normalise all fcs - zscore
    for i, fc in enumerate(fcs):
        fcs[i] = (fc - fc.mean(axis=0)) / fc.std(axis=0)
    
    # get training and test datasets
    train_data, test_data = train_test_split(fcs, test_size=0.2, random_state=42)

    return train_data, test_data

def get_avg_fc(fcs):
    corr_matrices = []

    for data in fcs:
        fc = np.corrcoef(data.T)
        corr_matrices.append(fc)

    stacked_corr = np.stack(corr_matrices, axis = 0)
    avg_corr = np.mean(stacked_corr, axis =0)
    
    return avg_corr

def get_avg_fmr(fmris):
    stacked_corr = np.stack(fmris, axis = 0)
    avg_corr = np.mean(stacked_corr, axis =0)
    
    return avg_corr

def train_model(fc_emp_train, fc_emp_test, num_epochs, learning_rate, bAB_E, sAB_E, bt_E, st_E, bAB_I, sAB_I, ts_length, node_size=100):
    params = ParamsRWWABT(bAB_E=par(val=bAB_E, fit_par=True), sAB_E=par(val=sAB_E, fit_par=True), bt_E=par(val=bt_E, fit_par=True),
                      st_E=par(val=st_E, fit_par=True), bAB_I=par(val=bAB_I, fit_par=True), sAB_I=par(val=sAB_I, fit_par=True))
    
    TPperWindow = 20
    step_size = 0.1
    repeat_size = 5

    # call model want to fit
    model = RNNRWWABT(node_size, TPperWindow, step_size, repeat_size, tr, sc, abeta, tau, use_fit_gains=True, params=params)

    # create objective function
    ObjFun = CostsRWW(model)

    # call model fit
    F = Model_fitting(model, ObjFun)

    # Train on training set
    F.train(u = 0, empFcs= [fc_emp_train], num_epochs = num_epochs, num_windows = int(ts_length / TPperWindow), learningrate = learning_rate)

    # Test on test set
    _, fc_sim = F.simulate(u = 0, num_windows=int(ts_length/TPperWindow), base_window_num = 20)

    fc_cor = F.evaluate(empFcs=[fc_emp_test], fc_sims=[fc_sim])

    return fc_cor

def objective(trial):
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.05, 0.1])
    num_epochs = trial.suggest_categorical('num_epochs', [20, 50, 100])
    step_size = trial.suggest_categorical('step_size', [0.01, 0.05, 0.1])
    g = trial.suggest_categorical('G', [20, 50, 80, 100, 500, 1000])
    tp_per_window = trial.suggest_categorical('TPperWindow', [5, 10, 20, 50])

    score = train_model(fc_emp_train=torch.from_numpy(fc_emp_train), fc_emp_test=fc_emp_test, g=g, num_epochs=num_epochs, ts_length=ts_length, learning_rate=learning_rate)
    return score

if __name__ == '__main__':
    study_name = sys.argv[1]
    # patient group
    pgroup = sys.argv[2]

    # assume that these files are in the directory we're working in
    ADNI_MERGE_PATH = 'ADNIMERGE_29Apr2024_wFiles.csv'
    SC_PATH = 'DTI_fiber_consensus_HCP.csv'
    ABETA_PATH = f'AB_{pgroup}.csv'
    TAU_PATH = f'TAU_{pgroup}.csv'

    optuna_trials = int(sys.argv[4])
    no_jobs = int(sys.argv[5])

    # GET ADDITIONAL DATA: SC, A-BETA and TAU
    # normalise structural connectivity matrix
    sc = np.genfromtxt(SC_PATH, delimiter=',')

    SC = (sc + sc.T) * 0.5
    sc = np.log1p(SC) / np.linalg.norm(np.log1p(SC))

    abeta_file = np.genfromtxt(ABETA_PATH, delimiter=",")
    abeta = torch.tensor(abeta_file, dtype=torch.float32)

    tau_file = np.genfromtxt(TAU_PATH, delimiter=",")
    tau = torch.tensor(tau_file, dtype=torch.float32)

    # get functional connectivity data
    train_data, test_data = get_data(ADNI_MERGE_PATH, pgroup=pgroup)

    tr = 0.75
    ts_length = train_data[0].shape[0]

    fc_emp_train = get_avg_fc(train_data)
    fc_emp_test = get_avg_fc(test_data)

    study = optuna.create_study(direction='maximize', study_name=study_name)  
    study.optimize(objective, n_trials=optuna_trials, n_jobs=no_jobs) 

    res = {
        'params': study.best_params,
        'score' : study.best_value
    }

    with open(f'optuna_{study.study_name}.json', "w") as file:
        json.dump(res, file)
    