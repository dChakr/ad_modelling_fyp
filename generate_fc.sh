#!/bin/bash

#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=1:mem=50gb:cpu_type=rome
#PBS -o /rds/general/user/dc420/home/ad_modelling_fyp/logs/
#PBS -e /rds/general/user/dc420/home/ad_modelling_fyp/logs/

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
pip install scikit-learn
pip install nibabel
pip install nilearn

cp $HOME/ad_modelling_fyp/data/Schaefer2018_116Parcels_7Networks_order_FSLMNI152_2mm.nii.gz $TMPDIR
cp $HOME/ad_modelling_fyp/generate_fc_matrix.py $TMPDIR
cp -r $HOME/FMRI_ADNI_DATA/$ARGUMENT/ $TMPDIR/fmri
mkdir $TMPDIR/fc

python generate_fc_matrix.py $TMPDIR/fmri $TMPDIR/fc $TMPDIR/Schaefer2018_116Parcels_7Networks_order_FSLMNI152_2mm.nii.gz

cp -r $TMPDIR/fc $HOME/FMRI_ADNI_DATA/fc