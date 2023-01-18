import sys
from utils import imread, imwrite
from oldschool import vessel_seg

if __name__ == "__main__":
    input_nifti_file = sys.argv[1]
    output_nifti_file = sys.argv[2]
    img_obj = imread(input_file)
    mask_obj = vessel_seg(img_obj)
    imwrite(output_nifti_file,mask_obj,use_compression=True):
