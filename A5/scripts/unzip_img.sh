#! /bin/bash

# unzip 
Project_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd)
img_unzip_tar="$Project_DIR/dataset/images"
msk_unzip_tar="$Project_DIR/dataset/masks"
mkdir -p $img_unzip_tar
mkdir -p $msk_unzip_tar

CCAgT_img_DIR="$Project_DIR/CCAgT/images/*.zip"
CCAgT_msk_DIR="$Project_DIR/CCAgT/masks/*.zip"
img_zip_arr=$(ls -a $CCAgT_img_DIR)
msk_zip_arr=$(ls -a $CCAgT_msk_DIR)

for path in ${img_zip_arr[@]}; do
    unzip $path -d $img_unzip_tar
done

for path in ${msk_zip_arr[@]}; do
    unzip $path -d $msk_unzip_tar
done