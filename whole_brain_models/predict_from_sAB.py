from whobpyt.datatypes import par
from whobpyt.models.RWWABT import RNNRWWABT, ParamsRWWABT
from whobpyt.optimization.custom_cost_RWW import CostsRWW
from whobpyt.run import Model_fitting
from sklearn.model_selection import train_test_split
import sys
import pickle

# array and pd stuff
import numpy as np
import pandas as pd
from scipy.io import loadmat # for reading in the .mat files
import torch
import json

import matplotlib.pyplot as plt

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

def set_up_model(sAB_E):
    node_size = 100
    TPperWindow = 20
    step_size = 0.1
    repeat_size = 5
    tr = 0.75

    bAB_E = 0.4320991563612215
    sAB_E = sAB_E
    bt_E = -0.6482380264178235
    st_E =  -2.7855677723437537
    bAB_I = -0.5
    sAB_I = -3
    
    params = ParamsRWWABT(bAB_E=par(val=bAB_E, fit_par=True), sAB_E=par(val=sAB_E, fit_par=True), bt_E=par(val=bt_E, fit_par=True),
                      st_E=par(val=st_E, fit_par=True), bAB_I=par(val=bAB_I, fit_par=True), sAB_I=par(val=sAB_I, fit_par=True))
    
    model = RNNRWWABT(node_size, TPperWindow, step_size, repeat_size, tr, sc, abeta, tau, True, params)
    ObjFun = CostsRWW(model)

    return model, ObjFun

def train_and_simulate(fc_emp_train, ts_length, model, ObjFun):
    num_epochs = 50
    TPperWindow = 20

    lr = 0.05
    sp_threshold = 31

    F = Model_fitting(model, ObjFun)
    F.train(u = 0, empFcs = [torch.from_numpy(fc_emp_train)], num_epochs = num_epochs, 
        num_windows = int(ts_length / TPperWindow), learningrate = lr, early_stopping=True, softplus_threshold = sp_threshold)
    
    _, fc_sim = F.simulate(u =0, num_windows=int(ts_length / TPperWindow), base_window_num=20)
    
    return fc_sim

def compute_fc_lower_triangle(fc, node_size=100):
    # Get the lower triangle
    mask_e = np.tril_indices(node_size, -1)
    lower_triangle = fc[mask_e]
    
    return lower_triangle

def plot_predictions(sAB_Es, prediced_vs, filename):
    plt.figure(figsize=(12,8))
    plt.plot(sAB_Es, prediced_vs)
    plt.title('Predicted Ventricular_ICV Value for CN Patient, Varying sAB_E')
    plt.xlabel('sAB_E Value')
    plt.ylabel('Ventricular_ICV')
    plt.savefig(filename)
    

if __name__ == '__main__':
    # assume that these files are in the directory we're working in (adapted to be)
    ADNI_MERGE_PATH = 'ADNIMERGE_29Apr2024_wFiles_mod.csv'
    SC_PATH = 'DTI_fiber_consensus_HCP.csv'
    ABETA_PATH = f'AB_CN.csv'
    TAU_PATH = f'TAU_CN.csv'
    PREDICTOR_PATH = 'gbregressor_ventricular_icv.pkl'   # '../predictors/gbregressor_ventricular_icv.pkl'

    with open(PREDICTOR_PATH, 'rb') as f:
        predictor = pickle.load(f)

    sc = np.genfromtxt(SC_PATH, delimiter=',')

    SC = (sc + sc.T) * 0.5
    sc = np.log1p(SC) / np.linalg.norm(np.log1p(SC))

    abeta_file = np.genfromtxt(ABETA_PATH, delimiter=",")
    abeta = torch.tensor(abeta_file, dtype=torch.float32)

    tau_file = np.genfromtxt(TAU_PATH, delimiter=",")
    tau = torch.tensor(tau_file, dtype=torch.float32)

    # get functional connectivity data
    train_data, test_data = get_data(ADNI_MERGE_PATH)
    
    ts_length = train_data[0].shape[0]
    fc_emp_train = get_avg_fc(train_data)
    fc_emp_test = get_avg_fc(test_data)

    sAB_Es = np.linspace(0.3, 4, 15)
    prediced_vs = []

    # -8, 9, 
    for sAB_E in sAB_Es:
        model, ObjFun = set_up_model(sAB_E)
        simulations = []

        # find the average accross sims
        for i in range(25):
            fc_sim = train_and_simulate(fc_emp_train, ts_length, model, ObjFun)
            lower_triangle_fc = compute_fc_lower_triangle(fc_sim)
            simulations.append(lower_triangle_fc)
            
        v = predictor.predict(simulations)
        prediced_vs.append(np.mean(v))
    
    prediced_vs = np.array(prediced_vs)

    np.savetxt('trialled_sAB_E.txt', sAB_Es)
    np.savetxt('predicted_ventr_icv_sAB_E.txt', prediced_vs)

    plot_predictions(sAB_Es, prediced_vs, 'predicted_ventr_icv_sAB_E_plot.png')