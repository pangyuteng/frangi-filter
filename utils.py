# copied from Satelle APIS
# https://github.com/sawtellellc/apis/blob/main/ct-scan-body-part-detector/ct-scan-body-part-detector.ipynb
#

import os
import requests
import zipfile
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt

def download_images(series_instance_uid,image_root="."):
    url = f"https://services.cancerimagingarchive.net/services/v4/TCIA/query/getImage?SeriesInstanceUID={series_instance_uid}"
    zip_file_path = os.path.join(image_root,series_instance_uid+'.zip')
    r = requests.get(url, allow_redirects=True)
    print(r.status_code)
    if r.status_code != 200:
        raise LookupError(f"ohoh {r.status_code}!")
    open(zip_file_path, 'wb').write(r.content)
    folder_path = os.path.join(image_root,series_instance_uid)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)
    return folder_path

def imread(myinput):
    if isinstance(myinput,list):
        file_list = myinput
        dicom_list = []
        for image_file in file_list:
            ds=pydicom.dcmread(image_file)
            dicom_list.append((ds.InstanceNumber,image_file))
            dicom_list = sorted(dicom_list,key=lambda x:x[0])
        dicom_names = [x[1] for x in dicom_list]
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_names)
    else:
        file_path = myinput
        reader= sitk.ImageFileReader()
        reader.SetFileName(file_path)
        
    img_obj = reader.Execute()
        
    return img_obj

def resample_img(itk_image, out_spacing):
    
    # Resample images to out_spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def imwrite(fpath,img_obj,use_compression=True):
    print(img_obj.GetSize())
    writer = sitk.ImageFileWriter()    
    writer.SetFileName(fpath)
    writer.SetUseCompression(use_compression)
    writer.Execute(img_obj)

def convert_to_nifti(folder_path,series_instance_uid,image_root=None):
    file_list = [os.path.join(folder_path,x) for x in os.listdir(folder_path)]
    img_obj = imread(file_list)
    nii_gz_path = os.path.join(image_root,f'{series_instance_uid}.nii.gz')
    imwrite(nii_gz_path,img_obj)
    return nii_gz_path

def prepare(series_instance_uid,image_root):
        
    os.makedirs(image_root,exist_ok=True)

    # download images from TCIA (TCIA sometimes is down, thus we opted for above s3 path)

    folder_path = download_images(series_instance_uid,image_root=image_root)
    nii_gz_path = convert_to_nifti(folder_path,series_instance_uid,image_root=image_root)

