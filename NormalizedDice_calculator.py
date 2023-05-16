## adapted from "https://gist.github.com/mingrui/5aa63ca498bbd615f932855c6a6dc724.js"

import nibabel as nib
import numpy as np
import os
import sys
import pandas as pd
import csv

def main(argv):
    csvFile_withPath = r"T:\MIP\Katie_Merriman\zhang_mri\ManualAIpaths2.csv"

    initial_results = []
    final_results = []
    r = 0
    df_csv = pd.read_csv(csvFile_withPath, sep=',', header=0)
    for rows, file_i in df_csv.iterrows():
        MouseID = str(file_i['MouseID'])
        pathManual = str(file_i['pathManual'])
        pathAI_UNET = str(file_i['pathAI_UNET'])
        print(MouseID)
        [dice, p, n, h] = test_nifti_all_labels_dice_score(MouseID, pathManual, pathAI_UNET)
        if dice == -1:
            continue
        initial_results.append([MouseID, dice, p, n, h])
        r = r + h

    r = r/len(initial_results)
    for mouse in initial_results:
        ID = mouse[0]
        DCE = mouse[1]
        p = mouse[2]
        n = mouse[3]
        h = mouse[4]
        k = h*(r**-1 - 1)
        nDCE = 2*(2 + k*p + n)**-1
        final_results.append([ID, DCE, nDCE, p, n, h, r, k])


    for i in range(len(final_results)):
        print(final_results[i])



def test_nifti_all_labels_dice_score(MouseID, pathManual, pathAI_UNET):
    manual_file_path = os.path.join(pathManual+'.gz')
    UNET_file_path = os.path.join(pathAI_UNET+'.gz')
    #manual_file_path = os.path.join(pathManual, 'id-'+MouseID+'_study_3_wk13.nii.gz')
    #UNET_file_path = os.path.join(pathAI_UNET, 'Session3-ZhangC_' + MouseID + '_mask.nii.gz')

    try:
        manual_nib = nib.load(manual_file_path)
        truth_data = manual_nib.get_data()
        UNET_nib = nib.load(UNET_file_path)
        seg_data = UNET_nib.get_data()
        area_sum = np.sum(seg_data) + np.sum(truth_data)
        if area_sum > 0:
            dice = np.sum(seg_data[truth_data > 0]) * 2.0 / area_sum
            tp = np.sum(seg_data[truth_data > 0])
            fp = np.sum(seg_data > 0) - tp
            fn = np.sum(truth_data > 0) - tp
            p = fp/tp
            n = fn/tp
            h = np.sum(seg_data > 0)/np.sum(seg_data == 0)

        else:
            dice = 1 # correctly predicted no tumor
            p = 0
            n = 0
            h = 0
        return [dice, p, n, h]

    except FileNotFoundError:
        print('No file for ', MouseID)
        dice = -1  # correctly predicted no tumor
        p = -1
        n = -1
        h = -1
        return [dice, p, n, h]




if __name__ == '__main__':
    main(sys.argv)
