'''
Script to generate FC matrices of all fMRI data files in the given direction
'''
import sys
import os
import numpy as np 

from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker

if __name__ == '__main__':
    SOURCE_DIR = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
    
    SCHAEFER_ATLAS_PATH_116 = 'data/Schaefer2018_116Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
    
    masker = NiftiLabelsMasker(
        labels_img=SCHAEFER_ATLAS_PATH_116,
        standardize="zscore_sample",
        standardize_confounds=True,
        high_variance_confounds=True,
        memory="nilearn_cache",
        verbose=2
    )
    
    correlation_measure = ConnectivityMeasure(
        kind="correlation",
        standardize="zscore_sample",
    )

    fmri_filenames = [file for file in os.listdir(SOURCE_DIR) if file.endswith('.nii.gz')]

    for file in fmri_filenames:
        
        fmri_data = os.path.join(SOURCE_DIR, file)
        time_series = masker.fit_transform(fmri_data)
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
        
        np.savetxt(f'{OUTPUT_DIR}/{file}_FC.csv', correlation_matrix, delimiter=",")

