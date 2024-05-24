#!/bin/bash
# Script to convert all the dicom PET scans into nifty files

parent_dir="/Users/dyutichakraborty/Library/CloudStorage/OneDrive-Personal/University/Year 4/FYP/data/adni/PET/ADNI_AV45"
output_dir="/Users/dyutichakraborty/Library/CloudStorage/OneDrive-Personal/University/Year 4/FYP/data/adni/PET/ADNI_AV45_OUT"

for sub_dir in "$parent_dir"/* ; do
  target_path=$(find "$sub_dir/AV45_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_6mm_Res" -mindepth 1 -maxdepth 1 -type d)
  echo $target_path

  if [ -d "$target_path" ]; then
    /Applications/MRIcroGL.app/Contents/Resources/dcm2niix -f "%n" -o "$output_dir" -p y -z y "$target_path"
  else
    echo "No unique subdirectory found in $sub_dir."
  fi
done