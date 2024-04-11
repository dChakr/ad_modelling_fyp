#!/bin/bash

#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=1:mem=60gb:cpu_type=rome
#PBS -o /rds/general/user/dc420/home/ad_modelling_fyp/logs/fmri_transfer/
#PBS -e /rds/general/user/dc420/home/ad_modelling_fyp/logs/fmri_transfer/

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
pip install scikit-learn
pip install nibabel

cp $HOME/ad_modelling_fyp/data/Schaefer2018_116Parcels_7Networks_order_FSLMNI152_2mm.nii.gz $TMPDIR
cp $HOME/ad_modelling_fyp/generate_fc_matrix.py $TMPDIR
cp $HOME/FMRI_ADNI_DATA/$ARGUMENT $TMPDIR
cp $HOME/ad_modelling_fyp/pwd.txt $TMPDIR

# cp -r $HOME/FMRI_ADNI_DATA/fmri/ $TMPDIR/fmri
mkdir $TMPDIR/fc
mkdir $TMPDIR/fmri

# Get the data
PASSWORD="(Fc'AF7sgR*!"
FILES_LIST="$TMPDIR/$ARGUMENT"

rsync -avvzP --progress --files-from="$FILES_LIST" bty184@login.hpc.qmul.ac.uk:/data/SBBS-PIDProject/ADNIfMRIPrepOut-Nov20-Clean-Parcellated/ADSP-PHC-Subjects/fc $TMPDIR/fmri

# Compute the fc matrices
python generate_fc_matrix.py $TMPDIR/fmri $TMPDIR/fc $TMPDIR/Schaefer2018_116Parcels_7Networks_order_FSLMNI152_2mm.nii.gz

cp -r $TMPDIR/fc $HOME/FMRI_ADNI_DATA/fc


