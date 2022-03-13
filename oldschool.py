import traceback
import imageio
import numpy as np
from scipy import ndimage
from skimage import measure
import SimpleITK as sitk

# naive lungseg using image processing methods.
def lung_seg(img_obj,kind=None,iterations=None):
    
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
    if kind == 'erode':
        lung_mask = ndimage.morphology.binary_erosion(lung_mask,iterations=iterations).astype(arr.dtype)
    else:
        pass

    lung_obj = sitk.GetImageFromArray(lung_mask.astype(arr.dtype))
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




def vessel_seg(img_obj):

    spacing = img_obj.GetSpacing()
    origin = img_obj.GetOrigin()
    direction = img_obj.GetDirection()
    
    # lungseg
    lung_obj = lung_seg(img_obj,kind='erode',iterations=1)
    lung_mask = sitk.GetArrayFromImage(lung_obj)

    arr_list = []
    for x in np.arange(0.5,3.5,1.0):
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(float(x))
        smoothed = gaussian.Execute(img_obj)
        myfilter = sitk.ObjectnessMeasureImageFilter()
        myfilter.SetBrightObject(True)
        myfilter.SetObjectDimension(1) # 1: lines (vessels),
        myfilter.SetAlpha(0.5) 
        myfilter.SetBeta(0.5)
        myfilter.SetGamma(5.0)
        tmp_obj = myfilter.Execute(smoothed)
        arr_list.append(sitk.GetArrayFromImage(tmp_obj))
    
    arr = np.max(np.array(arr_list),axis=0)
    arr[lung_mask==0]=0
    vessel_obj = sitk.GetImageFromArray(arr)
    vessel_obj.SetSpacing(spacing)
    vessel_obj.SetOrigin(origin)
    vessel_obj.SetDirection(direction)

    return vessel_obj


def fissure_seg(img_obj):

    spacing = img_obj.GetSpacing()
    origin = img_obj.GetOrigin()
    direction = img_obj.GetDirection()

    img = sitk.GetArrayFromImage(img_obj)
    # fissure val hovers around -600, setting th to be lower
    # so abnormal tissue >-200 will be enhanced at similar magnitude
    #img = img.clip(-1024,102)
    clipped_obj = sitk.GetImageFromArray(img)
    clipped_obj.SetSpacing(spacing)
    clipped_obj.SetOrigin(origin)
    clipped_obj.SetDirection(direction)

    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(float(0.7))
    smoothed = gaussian.Execute(clipped_obj)

    myfilter = sitk.ObjectnessMeasureImageFilter()
    myfilter.SetBrightObject(True)
    myfilter.SetObjectDimension(2) # 2: planes (plate-like structures)

    myfilter.SetAlpha(1.0)
    myfilter.SetBeta(1.0)
    myfilter.SetGamma(500.0)
    
    tmp_obj = myfilter.Execute(smoothed)
    arr = sitk.GetArrayFromImage(tmp_obj)

    lung_obj = lung_seg(img_obj,kind='erode',iterations=5)
    lung = sitk.GetArrayFromImage(lung_obj)

    arr[lung==0]=0
    fissure_obj = sitk.GetImageFromArray(arr)
    fissure_obj.SetSpacing(spacing)
    fissure_obj.SetOrigin(origin)
    fissure_obj.SetDirection(direction)

    return fissure_obj

# TODO: cythonize below
def fissure_seg_v2(img_obj):
    # https://github.com/pangyuteng/Lung-Lobes-Segmentation-in-CT-Scans/blob/docker/vector_region_growing.cxx
    raise NotImplementedError()

#           trachea     lung
# img:      dark    to  bright
# lung:     1       to  1
# tubeness: high    to  high
#
def is_airway(ind,prior_ind,img,lung,tube):

    try:        

        if np.take(lung,ind,mode='raise') == 0:
            return False

        # prior intensity not too far from current
        intensity = np.take(img,ind,mode='raise')
        pintensity = np.take(img,prior_ind,mode='raise')

        tubeness = np.take(tube,ind,mode='raise')
        ptubeness = np.take(tube,prior_ind,mode='raise')

        #print('img',intensity, pintensity, 'tube',tubeness,ptubeness)

        # if it is more or less air
        if np.abs(intensity-pintensity) < 5:
            return True
        
        # if it is more or less tube-like
        if np.abs(tubeness-ptubeness) < 5:
            return True

        if tubeness >  5:
            return True

    except IndexError:
        return False

    except:
        traceback.print_exc()

    return False

# TODO: needs to be in c++
def get_connected(ind,shape,search_radius=(2,2,2)):
    coord = np.unravel_index(ind,shape)
    Xs = coord[0]-search_radius[0]
    Xe = coord[0]+search_radius[0]+1 
    Ys = coord[1]-search_radius[1]
    Ye = coord[1]+search_radius[1]+1
    Zs = coord[2]-search_radius[2]
    Ze = coord[2]+search_radius[2]+1

    xs = np.arange(Xs,Xe,1)
    ys = np.arange(Ys,Ye,1)
    zs = np.arange(Zs,Ze,1)
    grid = np.meshgrid(xs,ys,zs)
    multi_index = np.array([grid[0].ravel(),grid[1].ravel(),grid[2].ravel()])
    region_ind = np.ravel_multi_index(multi_index,shape,mode='clip')
    return region_ind
    
# TODO: needs to be in c++
# reference https://stackoverflow.com/a/44143581/868736
def region_grow(img,lung,tubeness,seed_points):
    
    processed = np.zeros_like(img).astype(bool)
    outimg = np.zeros_like(img)
    
    while(len(seed_points) > 0):
        prior_ind = seed_points[0]
        for ind in get_connected(prior_ind, img.shape):
            if not np.take(processed,ind):
                if is_airway(ind,prior_ind,img,lung,tubeness):
                    np.put(outimg,ind,1)
                    if ind not in seed_points:
                        seed_points.append(ind)
                    np.put(processed,ind,True)
                    coord = np.unravel_index(ind,img.shape)
                    print(coord,len(seed_points))
        seed_points.pop(0)
        if np.sum(outimg)> 25000:
            break
    return outimg

def airway_seg(img_obj):
    
    img = sitk.GetArrayFromImage(img_obj)
    spacing = img_obj.GetSpacing()
    origin = img_obj.GetOrigin()
    direction = img_obj.GetDirection()

    bkgd = np.zeros(img.shape).astype(np.uint8)
    pad = 5
    bkgd[:,:,:pad]=1
    bkgd[:,:,-1*pad:]=1
    bkgd[:,:pad,:]=1
    bkgd[:,-1*pad:,:]=1
    # remove air in stomach
    bkgd[-1*pad:,:,:]=1
    

    # assume < -300 HU are voxels within lung
    procarr = (img < -300).astype(np.int)
    procarr = ndimage.morphology.binary_closing(procarr,iterations=1)

    label_image, num = ndimage.label(procarr)
    region = measure.regionprops(label_image)

    region = sorted(region,key=lambda x:x.area,reverse=True)
    lung_mask = np.zeros(img.shape).astype(np.uint8)
    
    # assume `x` largest air pockets except covering bkgd is lung, increase x for lung with fibrosis (?)
    for r in region[:3]:
        mask = label_image==r.label
        contain_bkgd = np.sum(mask*bkgd) > 0
        if contain_bkgd > 0:
            continue
        lung_mask[mask==1]=1
    
    # enhance tube like structure
    arr_list = []
    for x in np.arange(0.5,2.5,0.5):
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(float(x))
        smoothed = gaussian.Execute(img_obj)
        myfilter = sitk.ObjectnessMeasureImageFilter()
        myfilter.SetBrightObject(False)
        myfilter.SetObjectDimension(1)
        myfilter.SetAlpha(0.5) 
        myfilter.SetBeta(0.5)
        myfilter.SetGamma(5.0)
        tmp_obj = myfilter.Execute(smoothed)
        arr_list.append(sitk.GetArrayFromImage(tmp_obj))
    
    darktube = np.max(np.array(arr_list),axis=0)    
    darktube[lung_mask==0]=0
    
    # derive seed from top trachea
    trachea_mask = lung_mask.copy()
    trachea_mask[5:,:,:]=0
    label_image, num = ndimage.label(trachea_mask)
    region = measure.regionprops(label_image)
    region = sorted(region,key=lambda x:x.area,reverse=True)
    trachea_seed = np.array(region[0].centroid).astype(np.int)
    print(trachea_seed)
    seed_point = np.ravel_multi_index(trachea_seed,img.shape)
    print(seed_point)
    
    trachea_mask = region_grow(img,lung_mask,darktube,[seed_point])

    airway_obj = sitk.GetImageFromArray(trachea_mask)
    airway_obj.SetSpacing(spacing)
    airway_obj.SetOrigin(origin)
    airway_obj.SetDirection(direction)

    return airway_obj
