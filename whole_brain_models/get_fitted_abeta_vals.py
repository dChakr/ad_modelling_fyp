"""
Get the final fitted values of scaling and bias terms in the Abeta-Tau Model at the end of 
training with best set of hyperparameters, across the different patient groups
"""
from whobpyt.datatypes import par
from whobpyt.models.RWWABT import RNNRWWABT, ParamsRWWABT
from whobpyt.optimization.custom_cost_RWW import CostsRWW
from whobpyt.run import Model_fitting
from sklearn.model_selection import train_test_split
import sys

# array and pd stuff
import numpy as np
import pandas as pd
from scipy.io import loadmat # for reading in the .mat files
import torch
import json

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

def get_data(DATA_PATH, pgroup = 'CN'):
    patients = pd.read_csv(DATA_PATH)

    if pgroup == 'MCI':
        patients = patients[patients['DX_bl'].isin(['EMCI', 'LMCI'])]
    else:    
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

def get_hyperparams(hyperparam_file):
    with open(hyperparam_file) as file:
        hyperparams = json.load(file)
        
        hyperparams = hyperparams['params']
        return hyperparams

def set_up_model(hyperparams):
    node_size = 100
    TPperWindow = 20
    step_size = 0.1
    repeat_size = 5
    tr = 0.75

    bAB_E = hyperparams['bAB_E']
    sAB_E = hyperparams['sAB_E']
    bt_E = hyperparams['bt_E']
    st_E = hyperparams['st_E']
    bAB_I = hyperparams['bAB_I']
    sAB_I = hyperparams['sAB_I']
    
    params = ParamsRWWABT(bAB_E=par(val=bAB_E, fit_par=True), sAB_E=par(val=sAB_E, fit_par=True), bt_E=par(val=bt_E, fit_par=True),
                      st_E=par(val=st_E, fit_par=True), bAB_I=par(val=bAB_I, fit_par=True), sAB_I=par(val=sAB_I, fit_par=True))
    
    model = RNNRWWABT(node_size, TPperWindow, step_size, repeat_size, tr, sc, abeta, tau, True, params)
    ObjFun = CostsRWW(model)

    return model, ObjFun

def train(fc_emp_train, ts_length, hyperparams, model, ObjFun):
    num_epochs = 50
    TPperWindow = 20

    lr = hyperparams['learning_rate']
    sp_threshold = hyperparams['softplus_threshold']


    F = Model_fitting(model, ObjFun)
    F.train(u = 0, empFcs = [torch.from_numpy(fc_emp_train)], num_epochs = num_epochs, 
        num_windows = int(ts_length / TPperWindow), learningrate = lr, early_stopping=True, softplus_threshold = sp_threshold)
    
    return F.trainingStats.fit_params


if __name__ == '__main__':
    # patient group
    pgroup = sys.argv[1]

    # assume that these files are in the directory we're working in (adapted to be)
    ADNI_MERGE_PATH = 'ADNIMERGE_29Apr2024_wFiles_mod.csv'
    SC_PATH = 'DTI_fiber_consensus_HCP.csv'
    ABETA_PATH = f'AB_{pgroup}.csv'
    TAU_PATH = f'TAU_{pgroup}.csv'
    HYPERPARAM_PATH = f'optuna_{pgroup}_study_2.json'

    sc = np.genfromtxt(SC_PATH, delimiter=',')

    SC = (sc + sc.T) * 0.5
    sc = np.log1p(SC) / np.linalg.norm(np.log1p(SC))

    abeta_file = np.genfromtxt(ABETA_PATH, delimiter=",")
    abeta = torch.tensor(abeta_file, dtype=torch.float32)

    tau_file = np.genfromtxt(TAU_PATH, delimiter=",")
    tau = torch.tensor(tau_file, dtype=torch.float32)

    # get functional connectivity data
    train_data, test_data = get_data(ADNI_MERGE_PATH, pgroup=pgroup)
    
    ts_length = train_data[0].shape[0]
    fc_emp_train = get_avg_fc(train_data)
    fc_emp_test = get_avg_fc(test_data)

    hyperparams = get_hyperparams(HYPERPARAM_PATH)
    model, ObjFun = set_up_model(hyperparams)

    bAB_Es = np.zeros(20)
    sAB_Es = np.zeros(20)
    bAB_Is = np.zeros(20)
    sAB_Is = np.zeros(20)
    bt_Es = np.zeros(20)
    st_Es = np.zeros(20)


    for i in range(20):
        fitted_params = train(fc_emp_train, ts_length, hyperparams, model, ObjFun)
        
        bAB_Es[i] = (fitted_params['bAB_E'][-1])
        sAB_Es[i] = (fitted_params['sAB_E'][-1])
        bAB_Is[i] = (fitted_params['bAB_I'][-1])
        sAB_Is[i] = (fitted_params['sAB_I'][-1])
        bt_Es[i] = (fitted_params['bt_E'][-1])
        st_Es[i] = (fitted_params['st_E'][-1])
    
    np.savetxt(f'bAB_E_{pgroup}.txt', bAB_Es)
    np.savetxt(f'sAB_E_{pgroup}.txt', sAB_Es)
    np.savetxt(f'bAB_I_{pgroup}.txt', bAB_Is)
    np.savetxt(f'sAB_I_{pgroup}.txt', sAB_Is)
    np.savetxt(f'bt_E_{pgroup}.txt', bt_Es)
    np.savetxt(f'st_E_{pgroup}.txt', st_Es)