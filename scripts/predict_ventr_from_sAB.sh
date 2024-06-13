#!/bin/bash

#PBS -l walltime=09:00:00
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -o /rds/general/user/dc420/home/ad_modelling_fyp/logs/predicted_ventr/
#PBS -e /rds/general/user/dc420/home/ad_modelling_fyp/logs/predicted_ventr/

module load tools/prod
module load SciPy-bundle/2023.07-gfbf-2023a

cp -r $HOME/ad_modelling_fyp/whole_brain_models/whobpyt $TMPDIR
cp $HOME/ad_modelling_fyp/whole_brain_models/requirements.txt $TMPDIR

pip install -r requirements.txt

cp $HOME/ad_modelling_fyp/data/avg_scans/AB_CN.csv $TMPDIR
cp $HOME/ad_modelling_fyp/data/avg_scans/TAU_CN.csv $TMPDIR
cp $HOME/ad_modelling_fyp/predictors/gbregressor_ventricular_icv.pkl $TMPDIR
cp $HOME/ad_modelling_fyp/whole_brain_models/model/cn_abt_model.pkl $TMPDIR
cp $HOME/ad_modelling_fyp/whole_brain_models/model/AD_abt_model.pkl $TMPDIR
cp $HOME/ad_modelling_fyp/whole_brain_models/predict_from_sAB.py $TMPDIR

python predict_from_sAB.py 

cp trialled_sAB_E_and_I_AD_sc.txt $HOME/ad_modelling_fyp/data/prediction_res
cp predicted_ventr_icv_sAB_E_and_I_AD_sc.txt $HOME/ad_modelling_fyp/data/prediction_res
cp predicted_ventr_icv_sAB_E_and_I_AD_sc_plot.png $HOME/ad_modelling_fyp/data/prediction_res