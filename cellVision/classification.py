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

def getImageData(inputImagePath):
    # need to rewrite for the general case
    # set up image data, extract from tiff format.
    # extract multiple channels (8 page tiff -> 4 images, with green and red channels
    if '/home/ccbr_okraus/cell_segmentation/Segmentation_library/' not in sys.path:
        sys.path.append('/home/ccbr_okraus/cell_segmentation/Segmentation_library/')
    import Load_GR
    from PIL import Image
    
    num_frames = 4
    num_chan = 2
    max_w = 1300
    max_h = 1000
    
    data_out = np.zeros((num_frames,num_chan,max_h,max_w))

    im = Image.open(inputImagePath)
    green,red = Load_GR.load(im)
    green = Load_GR.convert(green)
    red = Load_GR.convert(red)

    for frame in range(num_frames):
        stacked_images = np.concatenate((red[frame][np.newaxis,:,:],green[frame][np.newaxis,:,:]),axis=0)
        mid = (green[0].shape[0]/2,green[0].shape[1]/2)
        data_out[frame,:,:,:] = stacked_images[:,mid[0] - max_h/2: mid[0] + max_h/2,
                                                       mid[1] - max_w/2: mid[1] + max_w/2] 
    return data_out

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

def _classify(path):
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

    #may change to better model (constatly training better networks)
    model_path = '/home/okraus/mil_models_backup/mil_models/Yeast_Protein_Localization/Yeast_NAND_a_10_scratch_Dropout_v5_MAP_early_stopping_best_model.npz'

    #load model and set evaluation type (MIL convolves across whole images)
    sizeX=1000
    sizeY=1300
    nn = modelEvalFunctions.loadResizedModel(model_path,sizeX,sizeY)
    model = modelEvalFunctions.evaluateModel_MIL(nn,localizationTerms,outputLayer='loc')

    curImages = getImageData(path)
    curImages = normalize_by_constant_values(curImages,norm_vals['means'],norm_vals['stdevs'])
    nn.ForwardProp({'X0':gnp.garray(curImages)})

    # GET RATIOS OF CLASSES
    #values of prediction maps above
    pred_maps = nn._layers['MIL_pool'].Z[0].as_numpy_array()
    #calculate relative activation of each map
    area = pred_maps.sum(1).sum(1) / pred_maps.sum()
    #calculate absolute area of each map (optional)
    area2 = pred_maps.sum(1).sum(1) / (pred_maps.shape[1]*pred_maps.shape[2])
    #plot relative activations per class, use area or area2
    area_lib = {}
    for i in range(len(localizationTerms)):
        area_lib[localizationTerms[i]] = area[i]
    gnp.free_reuse_cache()
    return area_lib, nn, curImages

#functions for segmentation

def getJacobian(nn_input,curImages_input):
    bfs =[i for i in nn_input.cfg.bfs()]
    bfs.reverse()
    #norm_vals = np.load('/home/morphology/mpg4/OrenKraus/Data_Sets/Yeast_Protein_Localization/Yolanda_Chong/overal_mean_std_for_single_cell_crops_based_on_Huh.npz')
    #curImages = normalize_by_constant_values(curImages_input,norm_vals['means'],norm_vals['stdevs'])
    gnp.free_reuse_cache()
    nn_input.ForwardProp({'X0':gnp.garray(curImages_input)})
    
    Z_new = nn_input._layers['MIL_pool'].Z.copy()

    _,delta = nn_input._layers['MIL_pool']._ComputeParamGradient(Z_new * nn_input._layers['MIL_pool'].A.reshape(4,17,1,1))
    for l in bfs[3:-1]:
        gnp.free_reuse_cache()
        delta = nn_input._layers[l].BackProp(gnp.relu(delta))
    out = delta.as_numpy_array()
    return out

#per class

def getJacobian_per_class(nn_input,curImages_input):
    out=np.zeros((4,17,2,1000,1300))
    
    bfs =[i for i in nn_input.cfg.bfs()]
    bfs.reverse()
    
    gnp.free_reuse_cache()
    nn_input.ForwardProp({'X0':gnp.garray(curImages_input)})
    
    Z_new = nn_input._layers['MIL_pool'].Z.copy()
    
    for loc in range(17):
        Z_new = nn_input._layers['MIL_pool'].Z.copy()
        Z_inv = gnp.zeros(Z_new.shape)
        Z_inv[:,loc,:,:] = Z_new[:,loc,:,:] * nn_input._layers['MIL_pool'].A[:,loc].reshape(4,1,1)
        _,delta = nn_input._layers['MIL_pool']._ComputeParamGradient(Z_inv)
        #_,delta = nn._layers['MIL_pool']._ComputeParamGradient(Z_new)
        for l in bfs[3:-1]:
            gnp.free_reuse_cache()
            delta = nn_input._layers[l].BackProp(gnp.relu(delta))
        out[:,loc,:,:,:] = delta.as_numpy_array()
    return out



def runLoopyBP_on_jacobians(jacobian_map_array, thresh_r = 0.1, thresh_g = 1):
    jabobian_stack = jacobian_map_array
    output_maps = np.zeros(jacobian_map_array.shape)

    for frame_num in range(len(jabobian_stack)):
        thresholded_jacobian = (np.int32(np.log(1+jabobian_stack[frame_num][0])>thresh_r)+\
                                np.int32(np.log(1+jabobian_stack[frame_num][1])>thresh_g))>0

        output_maps[frame_num] = loopyBP(thresholded_jacobian)
    return output_maps
        
def loopyBP(input_image,max_iterations=100,scale_visible_units=.1):
    output_image = np.zeros(input_image.shape)
    pred = np.zeros(input_image.shape)
    M,N = input_image.shape
    visible_messages = np.ones((2,M,N))
    edge_messages = np.ones((4,2,M,N))                            
    negative = np.float64(1 - input_image)
    positive = np.float64(input_image)

    normalize = np.exp(-(0-positive)**2) + np.exp(-(1-positive)**2)
    visible_messages[0] = np.exp(-scale_visible_units*(0-positive)**2) /normalize# + np.exp(-(0-positive)**2) 
    visible_messages[1] = np.exp(-scale_visible_units*(1-positive)**2) /normalize#np.exp(-(1-negative)**2) + 
    
    pred_old = 100*np.ones(pred.shape)
    
    for iteration in range(max_iterations):
        direction = 0 

        mask = range(4)
        mask.remove(direction+1)
        for j in range(0,N-1):
            prod_msg_0 = np.exp(np.sum(np.log(edge_messages[mask,0,:,j]),axis=0)) * visible_messages[0,:,j]
            prod_msg_1 = np.exp(np.sum(np.log(edge_messages[mask,1,:,j]),axis=0)) * visible_messages[1,:,j]

            normalize = costFunction(0,0) * prod_msg_0 + costFunction(1,0) * prod_msg_1 +\
                        costFunction(0,1) * prod_msg_0 + costFunction(1,1) * prod_msg_1
            edge_messages[direction,0,:,j+1] = (costFunction(0,0) * prod_msg_0 + costFunction(1,0) * prod_msg_1 )/normalize
            edge_messages[direction,1,:,j+1] = (costFunction(0,1) * prod_msg_0 + costFunction(1,1) * prod_msg_1 )/normalize

        direction = 1 

        mask = range(4)
        mask.remove(direction-1)   
        for j in range(N-1,0,-1):
            prod_msg_0 = np.exp(np.sum(np.log(edge_messages[mask,0,:,j]),axis=0)) * visible_messages[0,:,j]
            prod_msg_1 = np.exp(np.sum(np.log(edge_messages[mask,1,:,j]),axis=0)) * visible_messages[1,:,j] 

            normalize = costFunction(0,0) * prod_msg_0 + costFunction(1,0) * prod_msg_1 +\
                        costFunction(0,1) * prod_msg_0 + costFunction(1,1) * prod_msg_1
            edge_messages[direction,0,:,j-1] = (costFunction(0,0) * prod_msg_0 + costFunction(1,0) * prod_msg_1 )/normalize
            edge_messages[direction,1,:,j-1] = (costFunction(0,1) * prod_msg_0 + costFunction(1,1) * prod_msg_1 )/normalize

        direction = 2
        mask = range(4)
        mask.remove(direction+1) 

        for i in range(0,M-1):
            prod_msg_0 = np.exp(np.sum(np.log(edge_messages[mask,0,i,:]),axis=0)) * visible_messages[0,i,:]
            prod_msg_1 = np.exp(np.sum(np.log(edge_messages[mask,1,i,:]),axis=0)) * visible_messages[1,i,:]

            normalize = costFunction(0,0) * prod_msg_0 + costFunction(1,0) * prod_msg_1 +\
                        costFunction(0,1) * prod_msg_0 + costFunction(1,1) * prod_msg_1
            edge_messages[direction,0,i+1,:] = (costFunction(0,0) * prod_msg_0 + costFunction(1,0) * prod_msg_1 )/normalize
            edge_messages[direction,1,i+1,:] = (costFunction(0,1) * prod_msg_0 + costFunction(1,1) * prod_msg_1 )/normalize

        direction = 3 
        mask = range(4)
        mask.remove(direction-1) 

        for i in range(M-1,0,-1):

            prod_msg_0 = np.exp(np.sum(np.log(edge_messages[mask,0,i,:]),axis=0)) * visible_messages[0,i,:]
            prod_msg_1 = np.exp(np.sum(np.log(edge_messages[mask,1,i,:]),axis=0)) * visible_messages[1,i,:]

            normalize = costFunction(0,0) * prod_msg_0 + costFunction(1,0) * prod_msg_1 +\
                        costFunction(0,1) * prod_msg_0 + costFunction(1,1) * prod_msg_1
            edge_messages[direction,0,i-1,:] = (costFunction(0,0) * prod_msg_0 + costFunction(1,0) * prod_msg_1 )/normalize
            edge_messages[direction,1,i-1,:] = (costFunction(0,1) * prod_msg_0 + costFunction(1,1) * prod_msg_1 )/normalize


        normalize = (np.exp(np.sum(np.log(edge_messages),axis=0))*visible_messages).sum(0)
        pred  = np.exp(np.sum(np.log(edge_messages),axis=0))*visible_messages / normalize

        if np.abs(pred-pred_old).sum()<1e-4:
            print 'stopped at ',iteration,np.abs(pred-pred_old).sum()
            break
        else:
            print iteration,np.abs(pred-pred_old).sum()
            pred_old = pred
            
                  
    return pred
    
def costFunction(i,j):
    cost_mat = np.array([[0,1],[1,0]])
    return np.exp(-cost_mat[i,j])

