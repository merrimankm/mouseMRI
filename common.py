import os
import numpy as np
import SimpleITK as sitk
import uuid
import time
import sys

def ExposeMetaData(tags, key, defaultValue=None):
    if type(tags) != dict or key not in tags:
        return defaultValue
        
    if defaultValue is None:
        return tags[key]
    elif type(defaultValue) == list: # Probably means list of floats
        return [ float(value) for value in tags[key].strip().split("\\") ]
    elif type(defaultValue) == np.ndarray: # Probably means a list of floats
        return np.reshape([ float(value) for value in tags[key].strip().split("\\") ], defaultValue.shape)
    else:
        return type(defaultValue)(tags[key])
        
    return defaultValue

def GetImageOrientation(tags):
    direction3D = np.eye(3)
    direction3D[0:2, :] = ExposeMetaData(tags, "0020|0037", np.eye(3)[0:2,:])
    direction3D[2, :] = np.cross(direction3D[0,:], direction3D[1,:])
    
    return direction3D.transpose()
    
def GetImagePosition(tags):
    return ExposeMetaData(tags, "0020|0032", np.zeros([3]))
    
def GetSliceThickness(tags):
    if type(tags) == list:
        # Assumed sorted!
        
        if len(tags) < 2:
            return ExposeMetaData(tags[0], "0018|0050", 0.0)
            
        R = GetImageOrientation(tags[0])
        pos1 = GetImagePosition(tags[0])
        posN = GetImagePosition(tags[-1])
        
        return R.transpose().dot(posN-pos1)[2] / (len(tags)-1)

    return ExposeMetaData(tags, "0018|0050", 0.0)
    
def GetPixelSpacing(tags):
    return ExposeMetaData(tags, "0028|0030", np.zeros([2]))
    
def GetVoxelSpacing(tags):
    spacing3D = np.zeros([3])
    
    if type(tags) == list:
        spacing3D[0:2] = GetPixelSpacing(tags[0])
    else:
        spacing3D[0:2] = GetPixelSpacing(tags)

    spacing3D[2] = GetSliceThickness(tags)
    
    if spacing3D[2] <= 0.0:
        raise Exception(f"Calculated non-positive z spacing: z spacing = {spacing3D[2]}.")
    
    return spacing3D
    
def GetImageSize(tags):
    if type(tags) == dict:
        tags = [ tags ]

    size3D = np.zeros([3], dtype=np.int32)
    size3D[0] = ExposeMetaData(tags[0], "0028|0011", 0) # X = Columns
    size3D[1] = ExposeMetaData(tags[0], "0028|0010", 0) # Y = Rows
    size3D[2] = len(tags)
    
    return size3D

def GetMetaDataDictionaries(path, seriesUID = ""):
    if not os.path.exists(path):
        return None

    reader = sitk.ImageFileReader()
    reader.SetImageIO("GDCMImageIO")
        
    if os.path.isfile(path):
        reader.SetFileName(path)
        
        try:
            reader.ReadImageInformation()
            seriesUID = reader.GetMetaData("0020|000e").strip()
        except:
            return None
        
        path = os.path.dirname(path)
    
    if seriesUID is None or seriesUID == "":
        allSeriesUIDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
        
        if len(allSeriesUIDs) == 0:
            return None
        
        seriesUID = allSeriesUIDs[0]
        
    fileNames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, seriesUID)
    
    if len(fileNames) == 0: # What?
        return None
        
    allTags = []
    
    for fileName in fileNames:
        reader.SetFileName(fileName)
        try:
            reader.ReadImageInformation()
            
            tags = { key: reader.GetMetaData(key) for key in reader.GetMetaDataKeys() }
                
            allTags.append(tags)
        except:
            return None
            
    return allTags


def LoadDicomImage(path, seriesUID=None, dim=None, dtype=None):
    if not os.path.exists(path):
        return None

    reader2D = sitk.ImageFileReader()
    reader2D.SetImageIO("GDCMImageIO")
    reader2D.SetLoadPrivateTags(True)

    if dtype is not None:
        reader2D.SetOutputPixelType(dtype)

    if dim is None:  # Guess the dimension by the path
        dim = 2 if os.path.isfile(path) else 3

    if dim == 2:
        reader2D.SetFileName(path)

        try:
            return reader2D.Execute()
        except:
            return None

    if os.path.isfile(path):
        reader2D.SetFileName(path)

        try:
            reader2D.ReadImageInformation()
            seriesUID = reader2D.GetMetaData("0020|000e").strip()
        except:
            return None

        path = os.path.dirname(path)

    fileNames = []

    if seriesUID is None or seriesUID == "":
        allSeriesUIDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)

        if len(allSeriesUIDs) == 0:
            return None

        for tmpUID in allSeriesUIDs:
            tmpFileNames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, tmpUID)

            if len(tmpFileNames) > len(fileNames):
                seriesUID = tmpUID
                fileNames = tmpFileNames  # Take largest series
    else:
        fileNames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, seriesUID)

    if len(fileNames) == 0:  # Huh?
        return None

    reader3D = sitk.ImageSeriesReader()
    reader3D.SetImageIO("GDCMImageIO")
    reader3D.SetFileNames(fileNames)
    reader3D.SetLoadPrivateTags(True)
    reader3D.SetMetaDataDictionaryArrayUpdate(True)

    # reader3D.SetOutputPixelType(sitk.sitkUInt16)

    if dtype is not None:
        reader3D.SetOutputPixelType(dtype)

    try:
        image = reader3D.Execute()
    except:
        return None

    # Check if meta data is available!
    # Copy it if it is not!
    if not image.HasMetaDataKey("0020|000e"):
        for key in reader3D.GetMetaDataKeys(1):
            image.SetMetaData(key, reader3D.GetMetaData(1, key))

    return image


def SaveDicomImage(image, path, compress=True):
    # Implement pydicom's behavior
    def GenerateUID(prefix="1.2.826.0.1.3680043.8.498."):
        if not prefix:
            prefix = "2.25."

        return str(prefix) + str(uuid.uuid4().int)

    if image.GetDimension() != 2 and image.GetDimension() != 3:
        raise RuntimeError("Only 2D or 3D images are supported.")

    if not image.HasMetaDataKey("0020|000e"):
        print("Error: Reference meta data does not appear to be DICOM?", file=sys.stderr)
        return False

    writer = sitk.ImageFileWriter()
    writer.SetImageIO("GDCMImageIO")
    writer.SetKeepOriginalImageUID(True)
    writer.SetUseCompression(compress)

    newSeriesUID = GenerateUID()

    if image.GetDimension() == 2:
        writer.SetFileName(path)

        imageSlice = sitk.Image([image.GetSize()[0], image.GetSize()[1], 1], image.GetPixelID(),
                                image.GetNumberOfComponentsPerPixel())
        imageSlice.SetSpacing(image.GetSpacing())

        imageSlice[:, :, 0] = image[:]

        # Copy meta data
        for key in image.GetMetaDataKeys():
            imageSlice.SetMetaData(key, image.GetMetaData(key))

        newSopInstanceUID = GenerateUID()

        imageSlice.SetMetaData("0020|000e", newSeriesUID)
        imageSlice.SetMetaData("0008|0018", newSopInstanceUID)
        imageSlice.SetMetaData("0008|0003", newSopInstanceUID)

        try:
            writer.Execute(image)
        except:
            return False

        return True

    if not os.path.exists(path):
        os.makedirs(path)

    for z in range(image.GetDepth()):
        newSopInstanceUID = GenerateUID()

        """
        imageSlice = sitk.Image([image.GetSize()[0], image.GetSize()[1], 1], image.GetPixelID(),
                                image.GetNumberOfComponentsPerPixel())

        imageSlice[:] = image[:, :, z]

        imageSlice.SetSpacing(image.GetSpacing())
        """
        imageSlice = image[:,:,z]

        # Copy meta data
        for key in image.GetMetaDataKeys():
            imageSlice.SetMetaData(key, image.GetMetaData(key))

        # Then write new meta data ...
        imageSlice.SetMetaData("0020|000e", newSeriesUID)
        imageSlice.SetMetaData("0008|0018", newSopInstanceUID)
        imageSlice.SetMetaData("0008|0003", newSopInstanceUID)

        # Instance creation date and time
        imageSlice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        imageSlice.SetMetaData("0008|0013", time.strftime("%H%M%S"))

        # Image number
        imageSlice.SetMetaData("0020|0013", str(z + 1))

        position = image.TransformIndexToPhysicalPoint((0, 0, z))

        # Image position patient
        imageSlice.SetMetaData("0020|0032", f"{position[0]}\\{position[1]}\\{position[2]}")

        # Slice location
        imageSlice.SetMetaData("0020|1041", str(position[2]))

        # Spacing
        imageSlice.SetMetaData("0018|0050", str(image.GetSpacing()[2]))
        imageSlice.SetMetaData("0018|0088", str(image.GetSpacing()[2]))

        imageSlice.EraseMetaData("0028|0106")
        imageSlice.EraseMetaData("0028|0107")

        slicePath = os.path.join(path, f"{z + 1}.dcm")
        writer.SetFileName(slicePath)

        try:
            writer.Execute(imageSlice)
        except:
            print(f"Error: Failed to write slice '{slicePath}'.", file=sys.stderr)
            return False

    return True