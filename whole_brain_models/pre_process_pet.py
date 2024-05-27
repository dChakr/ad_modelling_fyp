# %%
import ants
import os
from nilearn import datasets, image
import numpy as np
import nibabel as nib

from nilearn.image import resample_to_img
from nilearn.input_data import NiftiLabelsMasker

def register_to_mni(input_file, output_file, mni_map):
    img = ants.image_read(input_file)
    mytx = ants.registration(fixed=mni_map , moving=img, type_of_transform='SyN' )
    warped_moving = mytx['warpedmovout']
    ants.image_write(warped_moving, output_file)

def get_parcellated_csv(input_file, output_file, schaefer_atlas):
    pet_img = nib.load(input_file)
    atlas_img = nib.load(schaefer_atlas)

    resampled_atlas_img = resample_to_img(atlas_img, pet_img, interpolation='nearest')
    masker = NiftiLabelsMasker(labels_img=resampled_atlas_img, standardize=True)

    region_signals = masker.fit_transform(pet_img)

    # normalise regional signals:
    region_signals = region_signals / np.sum(region_signals)

    # Save the extracted region signals to a CSV file
    np.savetxt(output_file, region_signals, delimiter=',')

def data_pipeline(input_dir, nifty_output_dir, res_output_dir, pet_type):
    mni_map = ants.image_read(ants.get_ants_data('mni'))
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, )
    schaefer_atlas = schaefer_atlas['maps']

    for filename in os.listdir(input_dir):
            if filename.endswith(".nii.gz"):
                filepath = os.path.join(input_dir, filename)
                register_to_mni(input_file=filepath, output_file=f'{nifty_output_dir}/{filename}', mni_map=mni_map)
    print('Begin parcellating:')
    for filename in os.listdir(nifty_output_dir):
            if filename.endswith(".nii.gz"):
                subject = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]
                print(f'Parcellating {pet_type} for sub {subject}')
                filepath = os.path.join(nifty_output_dir, filename)
                get_parcellated_csv(input_file=filepath, output_file=f'{res_output_dir}/{pet_type}_{subject}.csv', schaefer_atlas=schaefer_atlas)

if __name__ == '__main__':

    AB_INPUT_DIR = '../../data/adni/PET/ADNI_AV45_OUT'
    AB_OUTPUT_DIR = '../../data/adni/PET/ADNI_AV45_MNI'

    TAU_INPUT_DIR = '../../data/adni/PET/ADNI_AV1451_OUT'
    TAU_OUTPUT_DIR = '../../data/adni/PET/ADNI_AV1451_MNI'

    RES_DIR = '../../data/adni/PET/csv'

    data_pipeline(AB_INPUT_DIR, AB_OUTPUT_DIR, RES_DIR, "AB")
    data_pipeline(TAU_INPUT_DIR, TAU_OUTPUT_DIR, RES_DIR, "TAU")
