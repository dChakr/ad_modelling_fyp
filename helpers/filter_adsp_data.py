"""
Filter ADSP PHC Dataset and add subject files that we have
"""
import pandas as pd
import os

import re

def get_rid_viscode(filename):
    pattern = r'sub-ADNI\d+S(\d{4})_ses-(M\d{3})'
    match = re.search(pattern, filename)

    if match:
        rid = match.group(1)
        viscode = match.group(2)
        return rid, viscode        
    else:
        print("Pattern not found in the filename.")
        return None

def replace_viscode(str):
    if str == 'BL' or str == 'SC':
        return adsp_df['VISCODE2'].replace(str, 'M000')
    else:
        vis = str[1:]
        vis = vis.zfill(3)
        vis = 'M' + vis
        return adsp_df['VISCODE2'].replace(str, vis)


if __name__ == '__main__':

    ADSP_DATA_PATH = "data/ADSP_PHC_COGN_Dec2023_FILTERED.csv"
    FC_DATA_PATH = "../FMRI_ADNI_DATA/fc/"

    adsp_df = pd.read_csv(ADSP_DATA_PATH)
    adsp_df = adsp_df.drop(columns=adsp_df.columns[0])
    adsp_df = adsp_df.drop(columns=[
        'SUBJID', 'PHASE', 'VISCODE', 'EXAMDATE', 'PHC_Visit', 'PHC_Sex', 'PHC_Education', 'PHC_Ethnicity', 'PHC_Race', 'PHC_Age_Cognition', 
        'PHC_MEM_SE', 'PHC_MEM_PreciseFilter', 'PHC_EXF_SE', 'PHC_EXF_PreciseFilter', 'PHC_LAN_SE', 'PHC_LAN_PreciseFilter', 'PHC_VSP_SE',
        'PHC_VSP_PreciseFilter'
    ])

    adsp_df['VISCODE2'] = adsp_df['VISCODE2'].str.upper()

    # Pad the visit codes
    for val in adsp_df['VISCODE2'].unique():
        adsp_df['VISCODE2'] = replace_viscode(val)

    # Pad the RID values
    adsp_df['RID'] = adsp_df['RID'].apply(lambda x: str(x).zfill(4))

    adsp_df['FC_DATA'] = None

    fc_dir = os.listdir(FC_DATA_PATH)

    fc_files = [file for file in fc_dir if file.endswith('.mat')]

    for fc in fc_files:
        rid, viscode = get_rid_viscode(fc)
        adsp_df.loc[(adsp_df['RID'] == rid) & (adsp_df['VISCODE2'] == viscode), 'FC_DATA'] = fc
    
    adsp_df_filtered = adsp_df[adsp_df['FC_DATA'].notna()]
    adsp_df_filtered = adsp_df_filtered.drop(adsp_df_filtered[adsp_df_filtered['VISCODE2'] == 'M162'].index)
    adsp_df_filtered = adsp_df_filtered.drop(adsp_df_filtered[adsp_df_filtered['VISCODE2'] == 'M174'].index)
    adsp_df_filtered = adsp_df_filtered.drop(adsp_df_filtered[adsp_df_filtered['VISCODE2'] == 'M180'].index)
    adsp_df_filtered = adsp_df_filtered.drop(adsp_df_filtered[adsp_df_filtered['VISCODE2'] == 'M186'].index)
    adsp_df_filtered = adsp_df_filtered.drop(adsp_df_filtered[adsp_df_filtered['VISCODE2'] == 'M192'].index)
    
    adsp_df_filtered.to_csv('data/ADSP_PHC_COGN_Dec2023_FILTERED_wfiles_forTEMP.csv')