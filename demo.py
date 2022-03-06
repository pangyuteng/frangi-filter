import os
import SimpleITK as sitk
import numpy as np
import imageio

from utils import prepare, imread, resample_img
from PIL import Image
from oldschool import lung_seg, vessel_seg, airway_seg, fissure_seg

TARGET_SHAPE = [1,1,1]
if __name__ == "__main__":
    # download
    
    image_root = "images"
    series_instance_uid = '1.3.6.1.4.1.14519.5.2.1.6279.6001.113679818447732724990336702075'
    myfile = f'{image_root}/{series_instance_uid}.nii.gz'
    if not os.path.exists(myfile):
        prepare(series_instance_uid,image_root)

    # read    
    img_obj = imread(myfile)
    img_obj= resample_img(img_obj, TARGET_SHAPE)

    # lungseg
    lung_obj = lung_seg(img_obj)
    lung_mask = sitk.GetArrayFromImage(lung_obj)
    target_shape = (lung_mask.shape[1],lung_mask.shape[2])

    # visualize
    mip_list = []
    for x in range(3):
        img = np.sum(lung_mask,axis=x).squeeze()
        img = (255*(img-np.min(img))/(np.max(img)-np.min(img))).clip(0,255).astype(np.uint8)
        img = np.array(Image.fromarray(img).resize(size=target_shape))        
        mip_list.append(img)

    tmp = np.concatenate(mip_list,axis=1)
    imageio.imwrite(f'static/mip_lung.png',tmp)

    img = sitk.GetArrayFromImage(img_obj).clip(-1000,1000).astype(np.float)
    lung = sitk.GetArrayFromImage(lung_obj)
    # mask non-lung
    masked_img = img
    masked_img[lung==0] = -1000

    masked_img_obj = sitk.GetImageFromArray(masked_img)
    masked_img_obj.SetSpacing(img_obj.GetSpacing())
    masked_img_obj.SetOrigin(img_obj.GetOrigin())
    masked_img_obj.SetDirection(img_obj.GetDirection())

    img_obj = sitk.GetImageFromArray(img)
    img_obj.SetSpacing(img_obj.GetSpacing())
    img_obj.SetOrigin(img_obj.GetOrigin())
    img_obj.SetDirection(img_obj.GetDirection())

    for name,method,proc_obj in [
        ('airway',airway_seg,img_obj),
        ('vessel',vessel_seg,masked_img_obj),
        ('fissure',fissure_seg,masked_img_obj),
        ]:
        tmp_obj = method(proc_obj)

        # visualize
        tmp = sitk.GetArrayFromImage(tmp_obj)
        mip_list = []
        for x in range(3):
            img = np.sum(tmp,axis=x).squeeze()
            img = (255*(img-np.min(img))/(np.max(img)-np.min(img))).clip(0,255).astype(np.uint8)
            print(img.shape,target_shape)
            img = np.array(Image.fromarray(img).resize(size=target_shape))        
            mip_list.append(img)
        tmp = np.concatenate(mip_list,axis=1)
        imageio.imwrite(f'static/mip_{name}.png',tmp)
    
        



