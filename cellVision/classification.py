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

def getImageData(inputImagePath, frames, channels):
    # need to rewrite for the general case
    # set up image data, extract from tiff format.
    # extract multiple channels (8 page tiff -> 4 images, with green and red channels
    if '/home/ccbr_okraus/cell_segmentation/Segmentation_library/' not in sys.path:
        sys.path.append('/home/ccbr_okraus/cell_segmentation/Segmentation_library/')
    import Load_GR
    from PIL import Image
    im = Image.open(inputImagePath) 
    height = im.size[1]
    width = im.size[0]
    if height % 2:
        height-= 1
    if width % 2:
        width -= 1
    green,red = Load_GR.load(im)
    green = Load_GR.convert(green)
    red = Load_GR.convert(red)
    data_out = np.zeros((frames,channels,height,width))
    for frame in range(frames):
        stacked_images = np.concatenate((red[frame][np.newaxis,:,:],green[frame][np.newaxis,:,:]),axis=0)
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

#function for segmentation per class

def getJacobian_per_class(nn_input, loc):
    bfs =[i for i in nn_input.cfg.bfs()]
    bfs.reverse()
  
    Z_new = nn_input._layers['MIL_pool'].Z
    Z_inv = gnp.zeros(Z_new.shape)
    Z_inv[:,loc,:,:] = Z_new[:,loc,:,:] * nn_input._layers['MIL_pool'].A[:,loc].reshape(4,1,1)
    _,delta = nn_input._layers['MIL_pool']._ComputeParamGradient(Z_inv)
    for l in bfs[3:-1]:
        gnp.free_reuse_cache()
        delta = nn_input._layers[l].BackProp(gnp.relu(delta))
    return delta.as_numpy_array()

def _classify(path, name, frames, channels, target, choices):
    gnp.free_reuse_cache()
    #GPU TO USE, WE HAVE 2, I PREFER IF YOU'RE USING GPU 0
    #whole images take up a lot of memory so we need to coordinate this. 
    # if you're not using the notebook or a script make sure to shutdown or restart the notebook
    # you can use nvidia-smi in terminal to see what process are running on the GPU
    gnp._useGPUid = 1
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

    nn = modelEvalFunctions.loadResizedModel(model_path,sizeX,sizeY)
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
    for i in range(len(localizationTerms)):
        if localizationTerms[i] not in choices:
            continue
        area_lib[localizationTerms[i]] = area[i]
        jacobian_per_class = getJacobian_per_class(nn,i)[target-1,0]
        im2show = np.int8(np.log(1+jacobian_per_class)>1+np.int8(np.log(1+jacobian_per_class)>.5))>0
        im2show = mh.dilate(mh.dilate(mh.dilate(mh.erode(mh.erode(mh.erode(im2show>0))))))
        plt.imshow(im2show)
        loc = str(settings.MEDIA_ROOT + '/classes/' + name.split('.')[0]+str(i))
        save(loc)
    del nn
    del model
    gnp.free_reuse_cache()
    return area_lib
