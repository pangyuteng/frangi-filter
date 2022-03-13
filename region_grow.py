import traceback
import imageio
import numpy as np
from scipy import ndimage
from skimage import measure
import SimpleITK as sitk

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
        print(f'tubeness {ptubeness:4.2f}\t{tubeness:4.2f}\t intensity \t{pintensity:4d}\t{intensity:4d}\t')
        
        if intensity > -700:
            return False
            
        # if it is more or less air
        if np.abs(intensity-pintensity) < 5:
            return True
        
        # if it is more or less tube-like
        if np.abs(tubeness-ptubeness) < 5:
            return True

    except IndexError:
        return False

    except:
        traceback.print_exc()

    return False

    
# TODO: needs to be in c++
def get_connected(ind,shape,search_radius=(1,1,1)):
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
# + region grow like algo but with vector
# + graph cut
# + graph cnn
def region_grow(img,lung,tubeness,seed_points):
    
    raise NotImplementedError("in theory should work...")

    processed = np.zeros_like(img).astype(bool)
    outimg = np.zeros_like(img)
    
    while(len(seed_points) > 0):
        prior_ind,vec = seed_points[0]
        for ind in get_connected(prior_ind, img.shape):
            if not np.take(processed,ind):
                if is_airway(ind,prior_ind,img,lung,tubeness):
                    np.put(outimg,ind,1)
                    if ind not in seed_points:
                        seed_points.append(ind)
                    np.put(processed,ind,True)
                    coord = np.unravel_index(ind,img.shape)
                    #print(coord,len(seed_points),np.sum(outimg),50000)
        seed_points.pop(0)
        if np.sum(outimg)> 50000:
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
