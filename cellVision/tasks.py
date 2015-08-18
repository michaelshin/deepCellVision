from celery import task
import sys
if '/home/ccbr_okraus/cell_learning/' not in sys.path:
    sys.path.append('/home/ccbr_okraus/cell_learning')
if '/home/ccbr_okraus/cell_segmentation/Segmentation_library/' not in sys.path:
    sys.path.append('/home/ccbr_okraus/cell_segmentation/Segmentation_library/')
if '/home/michael/psinet/lib_python' not in sys.path: 
    sys.path.insert(0, '/home/michael/psinet/lib_python')
import modelEvalFunctions
import numpy as np
import gnumpy as gnp

from image_handler import save
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from django.conf import settings
import resource
import mahotas as mh
from django.core.mail import send_mail


def getImageData(inputImagePath, frames, channels):
     # need to rewrite for the general case
    # set up image data, extract from tiff format.
    # extract multiple channels (8 page tiff -> 4 images, with green and red channels
    from cellVision import Load
    from PIL import Image
    im = Image.open(inputImagePath) 
    height = im.size[1]
    width = im.size[0]
    if height % 2:
        height-= 1
    if width % 2:
        width -= 1
    data_out = np.zeros((frames,channels,height,width))
    if channels == 2:
        green,red = Load.load_GR(im)
        green = Load.convert(green)
        red = Load.convert(red)
        for frame in range(frames):
            stacked_images = np.concatenate((red[frame][np.newaxis,:,:],green[frame][np.newaxis,:,:]),axis=0)
            mid = (green[0].shape[0]/2,green[0].shape[1]/2)
            data_out[frame,:,:,:] = stacked_images[:,mid[0] - height/2: mid[0] + height/2,
                                                       mid[1] - width/2: mid[1] + width/2]
        return data_out, (height,width)
    if channels == 3:        
        green,red,far_red=Load.load_Red_FarRed(im)
        green = Load.convert(green)
        red = Load.convert(red)
        far_red = Load.convert(far_red)
        for frame in range(frames):
            stacked_images = np.concatenate((red[frame][np.newaxis,:,:],green[frame][np.newaxis,:,:],far_red[frame][np.newaxis,:,:]),axis=0)
            mid = (green[0].shape[0]/2,green[0].shape[1]/2)
            data_out[frame,:,:,:] = stacked_images[:,mid[0] - height/2: mid[0] + height/2,
                                                       mid[1] - width/2: mid[1] + width/2]
        return data_out, (height,width)

def normalize_by_constant_values(inputData,means,stds):
    # normalize data for NN
    if inputData.shape[1:] == means.shape:
        #normalize by whole images
        inputData = (inputData - means) / stds
        raise NameError('bad image configuration - should be tensor')
    else:
        #normalize by single values per channel
        for channel in range(len(means)):
            inputData[:,channel,:,:] = (inputData[:,channel,:,:] - means[channel]) / stds[channel]
    return inputData

#function for segmentation for whole image

def getJacobian(nn_input, frames):
    bfs =[i for i in nn_input.cfg.bfs()]
    bfs.reverse()
   
    Z_new = nn_input._layers['MIL_pool'].Z
    _,delta = nn_input._layers['MIL_pool']._ComputeParamGradient(Z_new * nn_input._layers['MIL_pool'].A.reshape(frames,17,1,1))
    for l in bfs[3:-1]:
        gnp.free_reuse_cache()
        delta = nn_input._layers[l].BackProp(gnp.relu(delta))
    return delta.as_numpy_array()

#function for segmentation per class

def getJacobian_per_class(nn_input, loc, frames):
    bfs =[i for i in nn_input.cfg.bfs()]
    bfs.reverse()
  
    Z_new = nn_input._layers['MIL_pool'].Z
    Z_inv = gnp.zeros(Z_new.shape)
    Z_inv[:,loc,:,:] = Z_new[:,loc,:,:] * nn_input._layers['MIL_pool'].A[:,loc].reshape(frames,1,1)
    _,delta = nn_input._layers['MIL_pool']._ComputeParamGradient(Z_inv)
    for l in bfs[3:-1]:
        gnp.free_reuse_cache()
        delta = nn_input._layers[l].BackProp(gnp.relu(delta))
    return delta.as_numpy_array()

#filter the segmentation
def mahotas_clean_up_seg(input_jacobian,frame_num):
    import mahotas as mh
    dsk = mh.disk(7)
    thresh_r = 0.1
    thresh_g = 1
    size_cutoff = 200
    
    thresholded_jacobian = (np.int32(np.log(1+input_jacobian[frame_num][0])>thresh_r)+\
                            np.int32(np.log(1+input_jacobian[frame_num][1])>thresh_g))>0
    thresholded_jacobian = mh.close_holes(thresholded_jacobian)
    thresholded_jacobian = mh.erode(thresholded_jacobian,dsk)
    labeled = mh.label(thresholded_jacobian)[0]
    sizes = mh.labeled.labeled_size(labeled)
    too_small = np.where(sizes < size_cutoff)
    labeled = mh.labeled.remove_regions(labeled, too_small)
    thresholded_jacobian = labeled>0
    thresholded_jacobian = mh.dilate(thresholded_jacobian,dsk)
    return thresholded_jacobian

#overlay ontop of the colour map
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def show_segmentation_boundaries(input_image,seg_mask,frame_num,sizeX,sizeY):
    import mahotas as mh
    cmap = plt.get_cmap('gray')
    #outlines
    dsk = mh.disk(7)
    boundaries = mh.dilate(seg_mask,dsk)-seg_mask
    boundaries=boundaries>0
    cmap_2_use = truncate_colormap(cmap,0.9,1)
    border2overlay = np.ma.masked_where(boundaries ==0, boundaries)

    x =mh.as_rgb(input_image[frame_num][0,:,:],input_image[frame_num][1,:,:],np.zeros((sizeY,sizeX)))
    x[:,:,1][x[:,:,1]>240]=240
    x[:,:,0][x[:,:,0]>240]=240

    slice2show = [slice(0,sizeY),slice(0,sizeX)]

    plt.imshow(x[slice2show]/np.array([240.,240.,1]).reshape(1,1,3))
    plt.imshow(border2overlay[slice2show],cmap_2_use,alpha=0.8),plt.axis('off')    

@task
def _classify(path, name, frames, channels, target, choices, CellObject):
    gnp.free_reuse_cache()
    #GPU TO USE, WE HAVE 2, I PREFER IF YOU'RE USING GPU 0
    #whole images take up a lot of memory so we need to coordinate this. 
    # if you're not using the notebook or a script make sure to shutdown or restart the notebook
    # you can use nvidia-smi in terminal to see what process are running on the GPU
    gnp._useGPUid = 0
    #protein localization categories
    localizationTerms=['ACTIN', 'BUDNECK', 'BUDTIP', 'CELLPERIPHERY', 'CYTOPLASM',
       'ENDOSOME', 'ER', 'GOLGI', 'MITOCHONDRIA', 'NUCLEARPERIPHERY',
       'NUCLEI', 'NUCLEOLUS', 'PEROXISOME', 'SPINDLE', 'SPINDLEPOLE',
       'VACUOLARMEMBRANE', 'VACUOLE']
    
    #normalization values (don't need to change)
    norm_vals = np.load('/home/morphology/mpg4/OrenKraus/Data_Sets/Yeast_Protein_Localization/Yolanda_Chong/overal_mean_std_for_single_cell_crops_based_on_Huh.npz')

    #may change to better model (constatly training bgnumpy.track_memory_usage=Trueetter networks)
    model_path = '/home/okraus/mil_models_backup/mil_models/Yeast_Protein_Localization/Yeast_NAND_a_10_scratch_Dropout_v5_MAP_early_stopping_best_model.npz'

    #load model and set evaluation type (MIL convolves across whole images)
    #change size
   

    curImages, sizes = getImageData(path, frames, channels)
    curImages = normalize_by_constant_values(curImages,norm_vals['means'],norm_vals['stdevs'])
    
    sizeX=sizes[1]
    sizeY=sizes[0]

    nn = modelEvalFunctions.loadResizedModel(model_path,sizeY,sizeX)
    model = modelEvalFunctions.evaluateModel_MIL(nn,localizationTerms,outputLayer='loc')

    
    nn.ForwardProp({'X0':gnp.garray(curImages)})

    # GET RATIOS OF CLASSES
    #values of prediction maps above
    pred_maps = nn._layers['MIL_pool'].Z[target-1].as_numpy_array()
    #calculate relative activation of each map
    area = pred_maps.sum(1).sum(1) / pred_maps.sum()
    #calculate absolute area of each map (optional)
    area2 = pred_maps.sum(1).sum(1) / (pred_maps.shape[1]*pred_maps.shape[2])
    #plot relative activations per class, use area or area2
    area_lib = {}

    jacobian = getJacobian(nn,frames)
    plt.imshow(jacobian[target-1,0])
    loc = str(settings.MEDIA_ROOT + '/classes/' + name.split('.')[0]+"_FULL0")
    save(loc)
    
    mahotas_segmentation = mahotas_clean_up_seg(jacobian,target-1)
    plt.imshow(mahotas_segmentation)
    loc = str(settings.MEDIA_ROOT + '/classes/' + name.split('.')[0]+"_FULL1")
    save(loc)

    show_segmentation_boundaries(curImages,mahotas_segmentation,target-1,sizeX, sizeY)
    loc = str(settings.MEDIA_ROOT + '/classes/' + name.split('.')[0]+"_FULL2")
    save(loc)

    top5indices = np.argsort(area)[::-1][:5]
    del jacobian
    del mahotas_segmentation

    for i in range(len(localizationTerms)):
        if i in top5indices:
            area_lib[localizationTerms[i]] = area[i]
            jacobian_per_class = getJacobian_per_class(nn,i,frames)[target-1,0]
            im2show = np.int8(np.log(1+jacobian_per_class)>1+np.int8(np.log(1   +jacobian_per_class)>.5))>0
            im2show = mh.dilate(mh.dilate(mh.dilate(mh.erode(mh.erode(mh.erode(im2show>0))))))
            plt.imshow(im2show)
            loc = str(settings.MEDIA_ROOT + '/classes/' + name.split('.')[0]+"_"+localizationTerms[i])
            save(loc)
            np.save(loc, im2show)
            continue
        if localizationTerms[i] not in choices:
            continue
        area_lib[localizationTerms[i]] = area[i]
        jacobian_per_class = getJacobian_per_class(nn,i,frames)[target-1,0]
        im2show = np.int8(np.log(1+jacobian_per_class)>1+np.int8(np.log(1+jacobian_per_class)>.5))>0
        im2show = mh.dilate(mh.dilate(mh.dilate(mh.erode(mh.erode(mh.erode(im2show>0))))))
        plt.imshow(im2show)
        loc = str(settings.MEDIA_ROOT + '/classes/' + name.split('.')[0]+"_"+localizationTerms[i])
        save(loc)
        np.save(loc, im2show) #save the array to give as a raw file
    del nn
    del model
    gnp.free_reuse_cache()
    f = [['Class', 'Area']]
    for key in area_lib:
        f.append([str(key), area_lib[key]])
    CellObject.activations = f
    CellObject.save()
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    for arr in f:
        ws.append(arr)
    wb.save(settings.MEDIA_ROOT + '/classes/' + name.split('.')[0] + '.xlsx')
    send_mail('Deep Cell Vision', 'Your image has been classified. Go to http://deepcellvision.com/results/' +CellObject.name + ' to see your results' , 'deepCellVision@gmail.com',
    [CellObject.email], fail_silently=False)
    return
