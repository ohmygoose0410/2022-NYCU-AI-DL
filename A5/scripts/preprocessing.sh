#!/bin/bash

if command -v python3 &> /dev/null
then
	python3 ./preprocessing.py \
        -orig_img_dir ./dataset/images \
        -orig_msk_dir ./dataset/masks \
        -save_json ./dataset/dataset.json \
        -save_samples ./dataset/samples
    
else
    echo "python3 could not be found~~"
fi

exit 0