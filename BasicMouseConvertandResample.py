
import os
import SimpleITK as sitk
import pandas as pd
import numpy as np



class DICOMtoNIFTI():
    def __init__(self):
        self.csv_file = r'T:\MIP\Katie_Merriman\zhang_mri\MouseQC.csv'
        self.saveFolder = r'T:\MIP\Katie_Merriman\zhang_mri\3MouseResampled'

    def startConversion(self):
        df_csv = pd.read_csv(self.csv_file, sep=',', header=0)
        for rows, file_i in df_csv.iterrows():
            mouse = (str(file_i['Mouse']))
            mouse = mouse[:-7]
            imgPath = (str(file_i['imgPath']))
            maskPath1 = (str(file_i['maskPath1']))
            maskPath2 = (str(file_i['maskPath2']))
            maskPath3 = (str(file_i['maskPath3']))

            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(imgPath)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            spacing = image.GetSpacing()
            if spacing[0] < 0.175: # ignores triple mice
                self.resample(image, mouse+".nii.gz")
            else:
                sitk.WriteImage(image, os.path.join(self.saveFolder, mouse+".nii.gz"))

            if not maskPath1 == 'nan': # if any masks exist
                mask1 = sitk.ReadImage(maskPath1)
                mask1arr = sitk.GetArrayFromImage(mask1)
                # tumor segmentation has different values for across different mice, but is always highest value in mask
                mask = np.where(mask1arr == np.ndarray.max(mask1arr), 1, 0)
                if not maskPath2 == 'nan':
                    mask2 = sitk.ReadImage(maskPath2)
                    mask2arr = sitk.GetArrayFromImage(mask2)
                    mask2arr = np.where(mask2arr == np.ndarray.max(mask2arr), 1, 0)
                    mask = mask + mask2arr
                if not maskPath3 == 'nan':
                    mask3 = sitk.ReadImage(maskPath3)
                    mask3arr = sitk.GetArrayFromImage(mask3)
                    mask3arr = np.where(mask3arr == np.ndarray.max(mask2arr), 1, 0)
                    mask = mask + mask3arr
                mask = mask.astype(float)
                maskImg = sitk.GetImageFromArray(mask) # SET AS FLOAT!
                maskImg.SetDirection(mask1.GetDirection())
                maskImg.SetSpacing(mask1.GetSpacing())
                maskImg.SetOrigin(mask1.GetOrigin())
                spacing = maskImg.GetSpacing()
                if spacing[0] < 0.175: # ignores triple mice
                    #the line commented out below saved a test image of the un-resampled image.
                        # Quality before resampling is great - quality issues occur only on resampling
                    #sitk.WriteImage(orig, os.path.join(self.saveFolder, mouse+"_maskUNRESAMPLED.nii.gz"))
                    self.resample(maskImg, mouse+"_mask.nii.gz")

                else:
                    sitk.WriteImage(maskImg, os.path.join(self.saveFolder, mouse+"_mask.nii.gz"))

    def resample(self, orig, name):
        resample = sitk.ResampleImageFilter()
        if "mask" in name:
            # I have tried 6 different options for SetInterpolator here,
            # some randomly, some from Google recommendations:
                # sitkLinear
                # sitkNearestNeighbor
                # sitkGaussian
                # sitkLabelGaussian
                # sitkSimilarity
                # stikBSpline
            # All produce the exact same output image
            resample.SetInterpolator = sitk.sitkNearestNeighbor
        else:
            resample.SetInterpolator = sitk.sitkLinear
        new_spacing = [0.1786, 0.1786, 0.5]
        orig_size = np.array(orig.GetSize(), dtype=int)
        orig_spacing = np.array(orig.GetSpacing())
        orig_size = [int(s) for s in orig_size]
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(int)  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_spacing)
        resample.SetOutputDirection(orig.GetDirection())
        resample.SetOutputOrigin(orig.GetOrigin())
        image_resamp = resample.Execute(orig)
        for meta_elem in orig.GetMetaDataKeys():
            image_resamp.SetMetaData(meta_elem, orig.GetMetaData(meta_elem))
        if "mask" in name:
            BinThreshImFilt = sitk.BinaryThresholdImageFilter()
            BinThreshImFilt.SetLowerThreshold(0.5)  # to be honest this is somewhat arbitrary
            BinThreshImFilt.SetOutsideValue(0)
            BinThreshImFilt.SetInsideValue(1)
            image_resamp = BinThreshImFilt.Execute(image_resamp)
        sitk.WriteImage(image_resamp, os.path.join(self.saveFolder, name))

if __name__ == '__main__':
    c = DICOMtoNIFTI()
    c.startConversion()


