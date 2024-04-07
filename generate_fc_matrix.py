#!/sw-eb/software/Python/3.10.4-GCCcore-11.3.0/bin/python
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
    SCHAEFER_ATLAS_PATH_116 = sys.argv[3]
    
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

    fmri_filenames = sorted([file for file in os.listdir(SOURCE_DIR) if file.endswith('.nii.gz')])

    for file in fmri_filenames:
        try:
            fmri_data = os.path.join(SOURCE_DIR, file)
            time_series = masker.fit_transform(fmri_data)
            correlation_matrix = correlation_measure.fit_transform([time_series])[0]
        except EOFError:
            print("EOFError: Compressed file ended before the end-of-stream marker was reached")
            pass
        else: 
            np.savetxt(f'{OUTPUT_DIR}/{file}_FC.csv', correlation_matrix, delimiter=",")
