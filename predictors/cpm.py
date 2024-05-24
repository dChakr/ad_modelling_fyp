# %% [markdown]
# # CPM Predictor 2
# 
# Following code from https://github.com/YaleMRRC/CPM/tree/master

# %%
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, t, ttest_1samp, pearsonr
from sklearn.linear_model import LinearRegression
from statsmodels.robust.robust_linear_model import RLM

import json
from joblib import dump

# %%
def compute_correlation(fmri):
    epsilon = 1e-6
    
    # calculate Pearson correlation between nodes and Fisher z-transform
    corr_matrix = np.corrcoef(fmri)
    
    epsilon = 1e-6  # Small constant to add to the diagonal - to prevent division by zero in the arctanh function
    clipped_corr_matrix = np.clip(corr_matrix + epsilon, -1 + epsilon, 1 - epsilon)
    fisher_z_matrix = np.arctanh(clipped_corr_matrix)
    
    return fisher_z_matrix

# %%
threshold = 0.01
def train_cpm(ipmat, pheno):

    """
    Accepts input matrices and pheno data
    Returns model
    @author: David O'Connor
    @documentation: Javid Dadashkarimi
    cpm: in cpm we select the most significant edges for subjects. so each subject
         have a pair set of edges with positive and negative correlation with behavioral subjects.
         It's important to keep both set in final regression task.  
    posedges: positive edges are a set of edges have positive
              correlatin with behavioral measures
    negedges: negative edges are a set of edges have negative
              correlation with behavioral measures
    """
    cc=[spearmanr(pheno,im) for im in ipmat]    
    
    rmat=np.array([c[0] for c in cc])
    pmat=np.array([c[1] for c in cc])
    rmat=np.reshape(rmat,[100,100])
    pmat=np.reshape(pmat,[100,100])
    posedges=(rmat > 0) & (pmat < threshold)    # edges correalated with higher ADAS Scores
    posedges=posedges.astype(int)
    negedges=(rmat < 0) & (pmat < threshold)    # edges correalated with lower ADAS Scores
    negedges=negedges.astype(int)
    pe=ipmat[posedges.flatten().astype(bool),:]
    ne=ipmat[negedges.flatten().astype(bool),:]
    pe=pe.sum(axis=0)/2
    ne=ne.sum(axis=0)/2


    if np.sum(pe) != 0:
        fit_pos=np.polyfit(pe,pheno,1)      # fit a curve through the positive edges
    else:
        fit_pos=[]

    if np.sum(ne) != 0:
        fit_neg=np.polyfit(ne,pheno,1)       # fit a curve through the negative edges
    else:
        fit_neg=[]

    return fit_pos,fit_neg,posedges,negedges

def run_validate(X,y,cvtype):
    """
    Accepts input matrices (X), phenotype data (y), and the type of cross-valdiation (cv_type)    
    Returns the R-values for positive model (Rpos), negative model (Rneg), and the combination
    X: the feature matrix of size (number of nodes x number of nodes x number of subjects)
    y: the phenotype vector of size (number of subjects)
    cv_type: the cross-valdiation type, takes one of the followings: 
    1) LOO: leave-one-out cross-validation
    2) 5k: 
    """
    numsubs=X.shape[2]
    X=np.reshape(X,[-1,numsubs])
    
    if cvtype == 'LOO':
        behav_pred_pos=np.zeros([numsubs])
        behav_pred_neg=np.zeros([numsubs])
        for loo in range(0,numsubs):

            print("Running LOO, sub no:",loo)
      
            train_mats=np.delete(X,[loo],axis=1)
            train_pheno=np.delete(y,[loo],axis=0)
            
            test_mat=X[:,loo]
            test_pheno=y[loo]

            pos_fit,neg_fit,posedges,negedges=train_cpm(train_mats,train_pheno)

            pe=np.sum(test_mat[posedges.flatten().astype(bool)])/2
            ne=np.sum(test_mat[negedges.flatten().astype(bool)])/2
            
            # Run model on test subject

            if len(pos_fit) > 0:
                behav_pred_pos[loo]=pos_fit[0]*pe + pos_fit[1]       # predict using the coefficients of the curve
            else:
                behav_pred_pos[loo]='nan'

            if len(neg_fit) > 0:
                behav_pred_neg[loo]=neg_fit[0]*ne + neg_fit[1]
            else:
                behav_pred_neg[loo]='nan'
          
        return behav_pred_pos, behav_pred_neg

        
#         Rpos=pearsonr(behav_pred_pos,y)[0]
#         Rneg=pearsonr(behav_pred_neg,y)[0]
        
#         return Rpos,Rneg
    

# %% [markdown]
# ### Get Features (FC Matrices)

# %%
def normalise_correlate_fc(fmri):
    fc_emp = fmri / np.max(fmri)
    fc_emp = np.corrcoef(fc_emp)
    return fc_emp

# %%
# ADAS_DATA = '../data/ADAS_ADNIGO23_17Apr2024_FILTERED_wfiles_197.csv'
PHC_DATA = '../data/ADSP_PHC_COGN_Dec2023_FILTERED_wfiles.csv'
df = pd.read_csv(PHC_DATA)
df = df.drop(columns=['RID', 'VISCODE2', 'PHC_Diagnosis', 'PHC_EXF', 'PHC_LAN', 'PHC_VSP'])
df.shape

# %%
# Get the FC data as numpy arrays
dim_x = len(df)
X = []

for i, file in enumerate(df['FC_DATA'].values[:50]):
    arr = loadmat(f'../{file}')['ROI_activity'][:100, :] # get the first 100 regions
    fc = normalise_correlate_fc(arr)
    
    # Add noise along the diagonal
    noise_level = 0.0001  # Adjust this value as needed
    noise = np.random.randn(100) * noise_level
    fc_with_noise = fc + np.diag(noise)
    
    X.append(fc_with_noise)

# %%
X = np.array(X)
X[0]

# %%
Y = df['PHC_MEM'][:50]
Y.shape

# %%
X.shape
X_transposed = np.transpose(X, (1, 2, 0))
X_transposed.shape

# %%
# results_df = pd.DataFrame(columns=['no_files', 'noise', 'fc_gen_method', 'pearson_r_pos', 'pearson_r_neg', 
#                                    'spearman_rho_pos', 'spearman_rho_neg', 'r2_pos', 'r2_neg', 'mse_pos',
#                                    'mse_neg'])
# results_df.head()

# %%
behav_pred_pos, behav_pred_neg = run_validate(X_transposed, Y, 'LOO')

res = {
    'no_files': 50, 
    'noise': 0.0001, 
    'fc_gen_method': 'normalise_correlate_fc', 
    'pearson_r_pos': pearsonr(behav_pred_pos,Y)[0], 
    'pearson_r_neg': pearsonr(behav_pred_neg,Y)[0],
    'spearman_rho_pos': spearmanr(behav_pred_pos,Y)[0], 
    'spearman_rho_neg': spearmanr(behav_pred_neg,Y)[0], 
    'r2_pos': r2_score(behav_pred_pos,Y), 
    'r2_neg': r2_score(behav_pred_neg,Y), 
    'mse_pos': mean_squared_error(behav_pred_pos,Y),
    'mse_neg': mean_squared_error(behav_pred_neg,Y)
}

# results_df = pd.concat([results_df, pd.DataFrame([res])])

# %%
results_df.head()

# %%
results_df.to_csv('../data/cpm_experiments.csv')

# %%
# results_df = results_df.drop(results_df.index[-2:])

# %%
# results_df = results_df.reset_index(drop=True)

# %%
# results_df.loc[2, 'fc_gen_method'] = 'normalise_correlate_fc'

# %% [markdown]
# ## Get Sample of Patients

# %%
# Define the number of buckets
n_buckets = 10

# Define the bins for splitting the column into buckets
bins = pd.cut(df['PHC_MEM'], bins=n_buckets, labels=False)

# %%
bins

# %%
# Group the DataFrame by the bins and create a dictionary of DataFrames
buckets = {i: group for i, group in df.groupby(bins)}

# %%
for i in range(n_buckets):
    bucket = buckets.get(i)
    if bucket is not None:
        print(f"Bucket {i}:")
        print(bucket.shape)

# %% [markdown]
# ### Stratified Sample

# %%
sample_df = None
total_sample = 0

for i in range(n_buckets):
    bucket = buckets.get(i)
    if bucket is not None:
        f = bucket.shape[0]
        sample_n = 1 if f * 0.1 < 1 else round(f * 0.1)
        print(sample_n)
        
        random = bucket.sample(sample_n)
        sample_df = pd.concat([sample_df, random])

sample_df = sample_df.reset_index(drop=True)
sample_df.shape

# %%
sample_df

# %%
dim_x = len(sample_df)
X_strat = []

for i, file in enumerate(sample_df['FC_DATA'].values):
    arr = loadmat(f'../{file}')['ROI_activity'][:100, :] # get the first 100 regions
    fc = compute_correlation(arr)
    
    # Add noise along the diagonal
    noise_level = 0.0001  # Adjust this value as needed
    noise = np.random.randn(100) * noise_level
    fc_with_noise = fc + np.diag(noise)
    
    X_strat.append(fc_with_noise)
X_strat = np.array(X_strat)

# %%
Y_strat = sample_df['PHC_MEM']
print(Y_strat.shape)

X_strat_transposed = np.transpose(X_strat, (1, 2, 0))
print(X_strat_transposed.shape)

# %%
behav_pred_pos, behav_pred_neg = run_validate(X_strat_transposed, Y_strat, 'LOO')

# %%
res = {
    'no_files': 134, 
    'noise': 0.0001, 
    'fc_gen_method': 'compute_correlation', 
#     'pearson_r_pos': pearsonr(behav_pred_pos,Y_strat)[0], 
    'pearson_r_neg': pearsonr(behav_pred_neg,Y_strat)[0],
    'spearman_rho_pos': spearmanr(behav_pred_pos,Y_strat)[0], 
#     'spearman_rho_neg': spearmanr(behav_pred_neg,Y_strat)[0], 
    'r2_pos': r2_score(behav_pred_pos,Y_strat), 
#     'r2_neg': r2_score(behav_pred_neg,Y_strat), 
    'mse_pos': mean_squared_error(behav_pred_pos,Y_strat),
#     'mse_neg': mean_squared_error(behav_pred_neg,Y_strat)
}

# results_df = pd.concat([results_df, pd.DataFrame([res])])

# %%
res

# %% [markdown]
# ### 'Equal' Sampling

# %%
# Want a sample of size 50 - get 6 from each bucket

sample_eq_df = None
sample_n = 5

for i in range(n_buckets):
    bucket = buckets.get(i)
    if bucket is not None:
        f = bucket.shape[0]
        bucket_n = sample_n if f > sample_n else f
        
        random = bucket.sample(bucket_n)
        sample_eq_df = pd.concat([sample_eq_df, random])

sample_eq_df = sample_eq_df.reset_index(drop=True)
sample_eq_df.shape

# %%
sample_eq_df.head()

# %%
dim_x = len(sample_eq_df)
X_eq = []

for i, file in enumerate(sample_eq_df['FC_DATA'].values):
    arr = loadmat(file)['ROI_activity'][:100, :] # get the first 100 regions
    fc = compute_correlation(arr)
    
    # Add noise along the diagonal
    noise_level = 0.0001  # Adjust this value as needed
    noise = np.random.randn(100) * noise_level
    fc_with_noise = fc + np.diag(noise)
    
    X_eq.append(fc_with_noise)
X_eq = np.array(X_eq)

# %%
Y_eq = sample_eq_df['TOTSCORE']
print(Y_eq.shape)

X_eq_transposed = np.transpose(X_eq, (1, 2, 0))
print(X_eq_transposed.shape)

# %%
behav_pred_pos, behav_pred_neg = run_validate(X_eq_transposed, Y_eq, 'LOO')

# %%
res = {
    'no_files': 39, 
    'noise': 0.0001, 
    'fc_gen_method': 'compute_correlation', 
    'pearson_r_pos': pearsonr(behav_pred_pos,Y_eq)[0], 
    'pearson_r_neg': pearsonr(behav_pred_neg,Y_eq)[0],
    'spearman_rho_pos': spearmanr(behav_pred_pos,Y_eq)[0], 
    'spearman_rho_neg': spearmanr(behav_pred_neg,Y_eq)[0], 
    'r2_pos': r2_score(behav_pred_pos,Y_eq), 
    'r2_neg': r2_score(behav_pred_neg,Y_eq), 
    'mse_pos': mean_squared_error(behav_pred_pos,Y_eq),
    'mse_neg': mean_squared_error(behav_pred_neg,Y_eq)
}
res

# %% [markdown]
# ### Try Larger Sample

# %%
ADAS_DATA_ALL = '../data/ADAS_ADNIGO23_17Apr2024_FILTERED_wfiles.csv'
df_all = pd.read_csv(ADAS_DATA_ALL)
df_all.shape

# %%
plt.figure(figsize=(8, 6))
plt.hist(df_all['TOTSCORE'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of ADAS-Cog Scores in Participants')
plt.xlabel('ADAS-Cog Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.axis('on')
# plt.savefig('../data/adas_score_dist_20.png')

plt.show()

# %%
# Define the number of buckets
n_buckets = 10

# Define the bins for splitting the column into buckets
bins = pd.cut(df_all['TOTSCORE'], bins=n_buckets, labels=False)

# %%
bins

# %%
buckets = {i: group for i, group in df_all.groupby(bins)}

for i in range(n_buckets):
    bucket = buckets.get(i)
    if bucket is not None:
        print(f"Bucket {i}:")
        print(bucket.shape)

# %%
# Want a sample of size 50 - get 5 from each bucket

# sample_eq_df = None
# sample_n = 6

# for i in range(n_buckets):
#     bucket = buckets.get(i)
#     if bucket is not None:
#         f = bucket.shape[0]
#         bucket_n = sample_n if f > sample_n else f
        
#         random = bucket.sample(bucket_n)
#         sample_eq_df = pd.concat([sample_eq_df, random])

# sample_eq_df = sample_eq_df.reset_index(drop=True)
# sample_eq_df.shape

# Stratified

sample_df = None
total_sample = 0

for i in range(n_buckets):
    bucket = buckets.get(i)
    if bucket is not None:
        f = bucket.shape[0]
        sample_n = 1 if f * 0.1 < 1 else round(f * 0.1)
        print(sample_n)
        
        random = bucket.sample(sample_n)
        sample_df = pd.concat([sample_df, random])

sample_df = sample_df.reset_index(drop=True)
sample_df.shape

# %%
# sample_eq_df

# %%
dim_x = len(sample_df)
X_strat = []

for i, file in enumerate(sample_df['FC_DATA'].values):
    arr = loadmat(file)['ROI_activity'][:100, :] # get the first 100 regions
    fc = compute_correlation(arr)
    
    # Add noise along the diagonal
    noise_level = 0.0001  # Adjust this value as needed
    noise = np.random.randn(100) * noise_level
    fc_with_noise = fc + np.diag(noise)
    
    X_strat.append(fc_with_noise)
X_strat = np.array(X_strat)

Y_strat = sample_df['TOTSCORE']
print(Y_strat.shape)

X_strat_transposed = np.transpose(X_strat, (1, 2, 0))
print(X_strat_transposed.shape)

# %%
behav_pred_pos, behav_pred_neg = run_validate(X_strat_transposed, Y_strat, 'LOO')

# %%
# print(behav_pred_neg)

# print(np.array(Y_strat))


res = {
    'no_files': 134, 
    'noise': 0.0001, 
    'fc_gen_method': 'compute_correlation', 
    'pearson_r_pos': pearsonr(behav_pred_pos,Y_strat)[0], 
#     'pearson_r_neg': pearsonr(behav_pred_neg,Y_strat)[0],
    'spearman_rho_pos': spearmanr(behav_pred_pos,Y_strat)[0], 
#     'spearman_rho_neg': spearmanr(behav_pred_neg,Y_strat)[0], 
    'r2_pos': r2_score(behav_pred_pos,Y_strat), 
#     'r2_neg': r2_score(behav_pred_neg,Y_strat), 
    'mse_pos': mean_squared_error(behav_pred_pos,Y_strat),
#     'mse_neg': mean_squared_error(behav_pred_neg,Y_strat)
}
res

# %%



