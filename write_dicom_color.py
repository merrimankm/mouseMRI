import SimpleITK as sitk
import numpy as np
from common import LoadDicomImage, SaveDicomImage
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os

def ColorizeImage(img):
    if img.GetNumberOfComponentsPerPixel() == 3:
        return img

    if img.GetNumberOfComponentsPerPixel() != 1:
        raise RuntimeError("Don't know how to colorize non-grayscale images.")

    npImg = sitk.GetArrayViewFromImage(img)

    # Window image
    minValue, maxValue = np.percentile(npImg, [5, 95])
    npImg = 255 * (npImg - minValue) / (maxValue - minValue)
    npImg = np.clip(npImg, 0, 255).astype(np.uint8)

    # Duplicate grayscale channel for R, G and B
    npImg = np.repeat(npImg[..., None], 3, axis=-1)  # R=G=B in colorized grayscale images

    newImg = sitk.GetImageFromArray(npImg, isVector=True)

    newImg.CopyInformation(img)

    # Copy DICOM tags!
    for key in img.GetMetaDataKeys():
        newImg.SetMetaData(key, img.GetMetaData(key))

    return newImg


def ColorizeMask(mask, colors):
    if mask.GetNumberOfComponentsPerPixel() == 3:
        return mask

    if mask.GetNumberOfComponentsPerPixel() != 1:
        raise RuntimeError("Don't know how to colorize non-grayscale masks.")

    npMask = sitk.GetArrayViewFromImage(mask)

    npNewMask = np.zeros(list(npMask.shape) + [3], dtype=np.uint8)
    npBlendMask = np.zeros(npMask.shape, dtype=np.uint8)

    # Get unique labels from mask and fill them in with colors
    for label in np.unique(npMask[npMask > 0]):
        if label not in colors:
            continue

        npBlendMask += (npMask == label)

        # There's probably a more efficient way to do this
        for c, color in enumerate(colors[label]):
            npNewMask[..., c] += ((npMask == label) * color).astype(np.uint8)

    newMask = sitk.GetImageFromArray(npNewMask, isVector=True)
    newMask.CopyInformation(mask)

    blendMask = sitk.GetImageFromArray(npBlendMask)  # Be consistent with returns

    return newMask, blendMask

def ColorizeProb(mask,image):
    npMask = sitk.GetArrayViewFromImage(mask)
    if np.max(npMask)>1:
        npMask = npMask/np.max(npMask)
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(npMask)
    rgb_img = np.delete(rgba_img, 3, 3)
    colimg = np.uint8(rgb_img * 255)
    newImage = sitk.GetImageFromArray(colimg)
    newImage.CopyInformation(image)

    # Copy DICOM tags!
    for key in image.GetMetaDataKeys():
        newImage.SetMetaData(key, image.GetMetaData(key))
    return newImage

def Blend(image, mask, blendMask, alpha=0.5):
    npImage = sitk.GetArrayFromImage(image)  # NOTE: Not "ArrayView"
    npMask = sitk.GetArrayViewFromImage(mask)
    npBlendMask = sitk.GetArrayViewFromImage(blendMask)

    npNewImage = npImage
    npNewImage[npBlendMask != 0] = (1.0 - alpha) * npImage[npBlendMask != 0] + alpha * npMask[npBlendMask != 0]

    newImage = sitk.GetImageFromArray(npNewImage.astype(np.uint8))

    newImage.CopyInformation(image)

    # Copy DICOM tags!
    for key in image.GetMetaDataKeys():
        newImage.SetMetaData(key, image.GetMetaData(key))

    return newImage

def BlendProbMap(image, blendProb, alpha=0.5):
    npImage = sitk.GetArrayFromImage(image)  # NOTE: Not "ArrayView"
    # npMask = sitk.GetArrayViewFromImage(mask)
    npBlendMask = sitk.GetArrayViewFromImage(blendProb)

    npNewImage = npImage
    npNewImage[npBlendMask != 0] = (1.0 - alpha) * npImage[npBlendMask != 0] + alpha * npBlendMask[npBlendMask != 0]

    newImage = sitk.GetImageFromArray(npNewImage.astype(np.uint8))

    newImage.CopyInformation(image)

    # Copy DICOM tags!
    for key in image.GetMetaDataKeys():
        newImage.SetMetaData(key, image.GetMetaData(key))

    return newImage

def runCase(dicomPath,probPath,binaryPath,savePath):
    colors = {
        1: [0, 0, 255],
        2: [255, 0, 0]
    }

    if not os.path.exists(savePath):
        os.mkdir(savePath)
        os.mkdir(savePath + "/prob_map")
        os.mkdir(savePath + "/segmentation")

    # convert probability map to DICOM

    imagePath = dicomPath #r"M:\Robert Huang\2022_07_01\other\Treated Prostate MRIs\DICOM\0648644\0JVJLDCX\4UMDUUX1"
    maskPath = probPath #r"M:\Robert Huang\2022_07_01\other\Treated Prostate MRIs\AI\0648644\output\lesion_prob.nii.gz"

    image = LoadDicomImage(imagePath)
    mask = sitk.ReadImage(maskPath)

    pimage = ColorizeProb(mask, image)

    image = ColorizeImage(image)
    # mask, blendMask = ColorizeMask(mask, colors)
    #
    # image = Blend(image, mask, blendMask, alpha=0.5)
    image = BlendProbMap(image, pimage, alpha=0.5)

    seriesNumber = "9999"
    # seriesDescription = image.GetMetaData("0008|103e") + " with overlaid segmentation"
    seriesDescription = "lesion detection probability map"
    derivationDescription = "probability"

    image.SetMetaData("0020|0011", str(seriesNumber))
    image.SetMetaData("0008|103e", seriesDescription)
    image.SetMetaData("0008|2111", derivationDescription)

    #SaveDicomImage(image, r"M:\Robert Huang\2022_07_01\other\Treated Prostate MRIs\AI\0648644\output_DICOM\prob_map",compress=True)
    SaveDicomImage(image, savePath + "/prob_map",compress=True)

    # convert combined mask to DICOM
    imagePath = dicomPath #r"M:\Robert Huang\2022_07_01\other\Treated Prostate MRIs\DICOM\0648644\0JVJLDCX\4UMDUUX1"
    maskPath = binaryPath#r"M:\Robert Huang\2022_07_01\other\Treated Prostate MRIs\AI\0648644\output\combined_lesion_mask.nii.gz"

    image = LoadDicomImage(imagePath)
    mask = sitk.ReadImage(maskPath)

    # pimage = ColorizeProb(mask,image)
    # sitk.WriteImage(pimage,"trythis.nii.gz")
    image = ColorizeImage(image)
    mask, blendMask = ColorizeMask(mask, colors)
    #
    image = Blend(image, mask, blendMask, alpha=0.5)
    # image = BlendProbMap(image,pimage,alpha=0.5)

    seriesNumber = "9998"
    seriesDescription = image.GetMetaData("0008|103e") + " with overlaid segmentation"
    # seriesDescription = "lesion detection probability map"
    derivationDescription = "wp + lesions segmentation"

    image.SetMetaData("0020|0011", str(seriesNumber))
    image.SetMetaData("0008|103e", seriesDescription)
    image.SetMetaData("0008|2111", derivationDescription)

    #SaveDicomImage(image,r"M:\Robert Huang\2022_07_01\other\Treated Prostate MRIs\AI\0648644\output_DICOM\segmentation",compress=True)
    SaveDicomImage(image, savePath + "/segmentation",compress=True)
    return

if __name__ == "__main__":
    # colors = {
    #     2: [0, 0, 255],
    #     3: [0, 255, 0],
    #     4: [255, 255, 0],
    #     5: [255, 0, 0]
    # }

    dcmpath = r"T:\MRIClinical\surgery_cases\3778666_20150827\dicoms\t2"
    probfile = r"T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data3\SURG-043\output\lesion_prob.nii.gz"
    maskfile = r"T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data3\SURG-043\output\mask_overlay.nii.gz"
    savepath = r"T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data3\SURG-043\dicoms"
    runCase(dicomPath=dcmpath, probPath=probfile, binaryPath=maskfile, savePath=savepath)
