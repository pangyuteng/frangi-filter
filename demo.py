import os
import SimpleITK as sitk
import numpy as np
import imageio

from utils import prepare, imread, resample_img
from PIL import Image
from oldschool import lung_seg, vessel_seg, airway_seg, fissure_seg

'''

class LungSegmenter(object):
    def __init__():
    def seg_airway():
    def seg_lung():
    def get_trachea():
    def get_lung_mask():
'''

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

    # resample
    img_obj= resample_img(img_obj, TARGET_SHAPE)

    # process...
    for name,method,args in [
        #('lung',lung_seg,(img_obj,)),
        ('airway',airway_seg,(img_obj,)),
        #('vessel',vessel_seg,(img_obj,)),
        #('fissure',fissure_seg,(img_obj,)),
        ]:
        print(f'generating {name}...')
        tmp_obj = method(*args)
        
        # visualize
        tmp = sitk.GetArrayFromImage(tmp_obj)
        target_shape = (tmp.shape[1],tmp.shape[2])
        mip_list = []
        for x in range(3):
            img = np.sum(tmp,axis=x).squeeze()
            img = (255*(img-np.min(img))/(np.max(img)-np.min(img))).clip(0,255).astype(np.uint8)
            img = np.array(Image.fromarray(img).resize(size=target_shape))        
            mip_list.append(img)
        
        png_file = f'static/mip_{name}.png'
        print(f'saving snapshot {png_file}')
        tmp = np.concatenate(mip_list,axis=1)
        imageio.imwrite(png_file,tmp)
    
        



