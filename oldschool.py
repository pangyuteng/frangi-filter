import numpy as np
from scipy import ndimage
from skimage import measure
import SimpleITK as sitk

# naive lungseg using image processing methods.
def lung_seg(img_obj):
    
    arr = sitk.GetArrayFromImage(img_obj)
    spacing = img_obj.GetSpacing()
    origin = img_obj.GetOrigin()
    direction = img_obj.GetDirection()  

    bkgd = np.zeros(arr.shape).astype(np.uint8)
    pad = 5
    bkgd[:,:,:pad]=1
    bkgd[:,:,-1*pad:]=1
    bkgd[:,:pad,:]=1
    bkgd[:,-1*pad:,:]=1
    
    # assume < -300 HU are voxels within lung
    procarr = (arr < -300).astype(np.int)
    procarr = ndimage.morphology.binary_closing(procarr,iterations=1)

    label_image, num = ndimage.label(procarr)
    region = measure.regionprops(label_image)

    region = sorted(region,key=lambda x:x.area,reverse=True)
    lung_mask = np.zeros(arr.shape).astype(np.uint8)
    
    # assume `x` largest air pockets except covering bkgd is lung, increase x for lung with fibrosis (?)
    x=2
    for r in region[:x]: # should just be 1 or 2, but getting x, since closing may not work.
        mask = label_image==r.label
        contain_bkgd = np.sum(mask*bkgd) > 0
        if contain_bkgd > 0:
            continue
        lung_mask[mask==1]=1
       
    lung_mask = ndimage.morphology.binary_closing(lung_mask,iterations=5)
    lung_mask = ndimage.morphology.binary_erosion(lung_mask,iterations=10).astype(arr.dtype)

    lung_obj = sitk.GetImageFromArray(lung_mask)
    lung_obj.SetSpacing(spacing)
    lung_obj.SetOrigin(origin)
    lung_obj.SetDirection(direction)

    return lung_obj

# sitk.ObjectnessMeasureImageFilter
# https://simpleitk.org/doxygen/latest/html/sitkObjectnessMeasureImageFilter_8h_source.html
# https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ObjectnessMeasureImageFilter.html
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.110.7722&rep=rep1&type=pdf
#   
#   ? 
#   https://github.com/InsightSoftwareConsortium/ITK/blob/f84720ee0823964bd135de8eb973acc40c1e70e1/Modules/Filtering/ImageFeature/include/itkHessianToObjectnessMeasureImageFilter.h 
#   https://github.com/InsightSoftwareConsortium/ITK/blob/f84720ee0823964bd135de8eb973acc40c1e70e1/Modules/Filtering/ImageFeature/include/itkHessianToObjectnessMeasureImageFilter.hxx
#
#   double       m_Alpha{ 0.5 };
#   double       m_Beta{ 0.5 };
#   double       m_Gamma{ 5.0 };
#   unsigned int m_ObjectDimension{ 1 };
#   bool         m_BrightObject{ true };
#   bool         m_ScaleObjectnessMeasure{ true };
#
#
#   /** Set/Get Alpha, the weight corresponding to R_A
#    * (the ratio of the smallest eigenvalue that has to be large to the larger ones).
#    * Smaller values lead to increased sensitivity to the object dimensionality. */
#   itkSetMacro(Alpha, double);
#   itkGetConstMacro(Alpha, double);

#   /** Set/Get Beta, the weight corresponding to R_B
#    * (the ratio of the largest eigenvalue that has to be small to the larger ones).
#    * Smaller values lead to increased sensitivity to the object dimensionality. */
#   itkSetMacro(Beta, double);
#   itkGetConstMacro(Beta, double);

#   /** Set/Get Gamma, the weight corresponding to S
#    * (the Frobenius norm of the Hessian matrix, or second-order structureness) */
#   itkSetMacro(Gamma, double);
#   itkGetConstMacro(Gamma, double);
#
def vessel_seg(masked_obj):
    myfilter = sitk.ObjectnessMeasureImageFilter()
    myfilter.SetBrightObject(True)
    myfilter.SetObjectDimension(1) # 1: lines (vessels),
    vessel_obj = myfilter.Execute(masked_obj)
    
    # TODO: need a loop for sigma
    # default values work surprisingly enhances the vessels in the sample image

    myfilter.SetAlpha(0.5)
    myfilter.SetBeta(0.5)
    myfilter.SetGamma(5.0)
    return vessel_obj

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6737148/pdf/10278_2018_Article_158.pdf
# alpha,beta,gamma: 0.05,10,100.0
# sigma: from 0.5 to 2.5 mm with a step size of 0.5 mm.
# TODO: need a loop for sigma
def airway_seg(img_obj):

    arr = sitk.GetArrayFromImage(img_obj)
    spacing = img_obj.GetSpacing()
    origin = img_obj.GetOrigin()
    direction = img_obj.GetDirection()  

    arr_list = []
    for sigma in np.arange(0.5,3,0.5):
        gaussian =sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(sigma)
        blurred = gaussian.Execute(img_obj)
        myfilter = sitk.ObjectnessMeasureImageFilter()
        myfilter.SetBrightObject(False)
        myfilter.SetObjectDimension(1) # 1: lines (vessels)
        myfilter.SetAlpha(0.05)
        myfilter.SetBeta(10.)
        myfilter.SetGamma(100.0)
        enhanced = myfilter.Execute(blurred)
        arr = sitk.GetArrayFromImage(enhanced)
        arr_list.append(arr)

    arr = np.array(arr_list)
    print(arr.shape)
    arr = np.max(arr,axis=0).squeeze()
    airway_obj = sitk.GetImageFromArray(arr)
    airway_obj.SetSpacing(spacing)
    airway_obj.SetOrigin(origin)
    airway_obj.SetDirection(direction)

    return airway_obj

def fissure_seg(masked_obj):
    myfilter = sitk.ObjectnessMeasureImageFilter()
    myfilter.SetBrightObject(True)
    myfilter.SetObjectDimension(2) # 2: planes (plate-like structures)
    # nope nope nope
    myfilter.SetAlpha(0.5) 
    myfilter.SetBeta(0.5)
    myfilter.SetGamma(1000.0)
    fissure_obj = myfilter.Execute(masked_obj)
    return fissure_obj
