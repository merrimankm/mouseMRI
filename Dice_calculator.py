## adapted from "https://gist.github.com/mingrui/5aa63ca498bbd615f932855c6a6dc724.js"

import nibabel as nib
import numpy as np
import os
import sys
import pandas as pd
import csv

def test_nifti_all_labels_dice_score(MouseID, pathManual, pathAI_ensemble, pathAI_UNET):
    #manual_file_path = os.path.join(pathManual,MouseID+'.nii.gz')
    manual_file_path = os.path.join(pathManual, 'id-' + MouseID + '_study_3_wk13_mask.nii.gz')
    ensemble_file_path = os.path.join(pathAI_ensemble,'Session3-ZhangC_'+MouseID+'_mask.nii.gz')
    UNET_file_path = os.path.join(pathAI_UNET, 'Session3-ZhangC_' + MouseID + '_mask.nii.gz')
    #manual_file_path = os.path.join(pathManual, 'gt_label_'+MouseID + '.nii.gz')
    #ensemble_file_path = os.path.join(pathAI_ensemble, 'gen_img_' + MouseID + '.nii.gz')
    #UNET_file_path = os.path.join(pathAI_UNET, 'gen_img_' + MouseID + '.nii.gz')
    try:
        manual_nib = nib.load(manual_file_path)
        manual_data = manual_nib.get_data()
        ensemble_nib = nib.load(ensemble_file_path)
        ensemble_data = ensemble_nib.get_data()
        UNET_nib = nib.load(UNET_file_path)
        UNET_data = UNET_nib.get_data()
        ensemble_dice = calculate_3Dvolume_all_labels_dice_score(ensemble_data, manual_data)
        UNET_dice = calculate_3Dvolume_all_labels_dice_score(UNET_data, manual_data)
        print(MouseID, ' ensemble dice score:', ensemble_dice, 'UNET dice score:', UNET_dice)
    except FileNotFoundError:
        print('No file for ', MouseID)


def calculate_nifti_all_labels_dice_score(seg_data, truth_data):
    z_range = range(seg_data.shape[-1])
    z_len = len(z_range)
    dice_sum = 0
    for z in  z_range:
        seg_slice = seg_data[:,:,z]
        truth_slice = truth_data[:,:,z]
        slice_dice = calculate_slice_all_labels_dice_score(seg_slice, truth_slice)
        dice_sum+=slice_dice

    return dice_sum / z_len

def calculate_slice_all_labels_dice_score(segmentation, truth):
    area_sum = np.sum(segmentation) + np.sum(truth)
    if area_sum > 0:
        return np.sum(segmentation[truth>0])*2.0 / area_sum
    else:
        return 1

def calculate_3Dvolume_all_labels_dice_score(seg_data, truth_data):
    area_sum = np.sum(seg_data) + np.sum(truth_data)
    if area_sum > 0:
        return np.sum(seg_data[truth_data > 0]) * 2.0 / area_sum
    else:
        return 1

def calculate_slice_one_label_dice_score(segmentation, truth, k):
    return np.sum(segmentation[truth == k]) * 2.0 / (np.sum(segmentation) + np.sum(truth))



def main(argv):
    csvFile_withPath = r"T:\MIP\Katie_Merriman\zhang_mri\ManualAIpaths.csv"
    #csvFile_withPath = r"T:\MIP\Katie_Merriman\zhang_mri\ManualAIpathsTest.csv"

    patientFolder = r"T:\MRIClinical\surgery_cases"
    saveFolder = r"T:\MIP\Katie_Merriman\Project2Data"
    df_csv = pd.read_csv(csvFile_withPath, sep=',', header=0)
    for rows, file_i in df_csv.iterrows():
        MouseID = str(file_i['MouseID'])
        pathManual = str(file_i['pathManual'])
        pathAI_ensemble = str(file_i['pathAI_ensemble'])
        pathAI_UNET = str(file_i['pathAI_UNET'])
        test_nifti_all_labels_dice_score(MouseID, pathManual, pathAI_ensemble, pathAI_UNET)


if __name__ == '__main__':
    main(sys.argv)
