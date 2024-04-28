'''
PRE-PROCESSING ADSP-PHC DATA

Filter the cognitive data from the ADSP Phenotype Harmonization Consortium to match subjects with those for whome we have function connectivity fMRI data. 
'''
import pandas as pd

def check_correct_patients(filtered_rows, subject_rids):
    ''' Check no required subject has been left out after filtering the ADSP-PHC data '''
    filtered_corr = True

    for rid in filtered_rows['RID'].values:
        if rid not in subject_rids:
            filtered_corr = filtered_corr and False
    return filtered_corr

def find_rid_from_fc(SUBJECT_LIST):
    # Find the RIDs for subjects that have functional connectivity data
    with open(SUBJECT_LIST, 'r') as file:
        subject_rids = [int(name[-4:]) for name in [line.strip() for line in file.readlines()]]
        # subject_rids = [int(line.strip()) for line in file.readlines()]

    # print(sorted(subject_rids))
    return subject_rids

def filter_adsp_data(ADSP_DIR, DATA_PATH, subject_rids):
    df = pd.read_csv(f'{ADSP_DIR}/{DATA_PATH}')

    # Remove subjects from the ADSP-PHC that don't have matching functional connectivity data
    idxs_to_drop = []
    for idx, rid in enumerate(df['RID'].values):
        if rid not in subject_rids:
            idxs_to_drop.append(idx)
            
    filtered_rows = df.drop(idxs_to_drop)

    if not check_correct_patients(filtered_rows, subject_rids):
        print('ERROR: incorrect filtering of ADSP-PHC data')
    return filtered_rows

def save_filtered_fc(output_file_subj, output_file_data, subject_rids, filtered_rows):
    
    # Save subject_list as those that we have cognitive scores for
    for rid in subject_rids:
        if rid not in filtered_rows['RID'].values:
            subject_rids.remove(rid)

    fc_subj = {'rid' : subject_rids}
    fc_subj_file = pd.DataFrame(fc_subj)
    fc_subj_file.to_csv(output_file_subj)
    filtered_rows.to_csv(output_file_data)


if __name__ == '__main__':

    ADSP_DIR = "data"
    DATA_PATH = "ADAS_ADNIGO23_17Apr2024.csv"
    SUBJECT_LIST = "../subjects.txt"  # subject RIDs from the FC data
    FILTERED_SUBJ_LIST = "filtered_fc_subjects.txt"
    FILTERED_ADSP_DATA = "ADAS_SCORES_FILTERED.csv"
    
    subject_rids = find_rid_from_fc(SUBJECT_LIST)
    filtered_rows = filter_adsp_data(ADSP_DIR, DATA_PATH, subject_rids)

    save_filtered_fc(FILTERED_SUBJ_LIST, FILTERED_ADSP_DATA, subject_rids, filtered_rows)

