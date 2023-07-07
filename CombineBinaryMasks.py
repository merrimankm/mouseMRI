import SimpleITK as sitk
import numpy as np
from common import LoadDicomImage, SaveDicomImage
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os

lesionPath = r"T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data3\SURG-043\output\lesion_mask.nii.gz"
prostPath = r"T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data3\SURG-043\wp_bt_undilated.nii.gz"


prostImg = sitk.ReadImage(prostPath)
lesionImg = sitk.ReadImage(lesionPath)

prost = sitk.GetArrayFromImage(prostImg)
lesion = sitk.GetArrayFromImage(lesionImg)

mask = np.where(lesion == 1, 2, prost)
maskImg = sitk.GetImageFromArray(mask)
maskImg.SetSpacing(prostImg.GetSpacing())
maskImg.SetDirection(prostImg.GetDirection())
maskImg.SetOrigin(prostImg.GetOrigin())


maskPath = r"T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data3\SURG-043\output\mask_overlay.nii.gz"
sitk.WriteImage(maskImg, maskPath)