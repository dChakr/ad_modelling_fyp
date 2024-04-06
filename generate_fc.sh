#!/bin/bash
batch_size=100
total_fmri_files=2017

num_batches=$(( (total_fmri_files + batch_size - 1) / batch_size ))

for ((batch = 2; batch <= num_batches; batch++))
do  
    start=$(( batch * (batch_size) ))
    # find . -type f | sort | head -n 200 | tail -n 100
    if [ $start -ge $total_fmri_files ]; then
        files_left=$(( total_fmri_files - (start - batch_size) ))
        echo "Last batch: tail -n $files_left"
    else
        echo "Current batch: head -n $start | tail -n $batch_size"
    fi
done