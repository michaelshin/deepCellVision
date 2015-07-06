def _segment(cell):
    # takes a numpy array of a microscopy
    # segments it based on filtering the image then applying a distance transform and
    # a watershed method to get the proper segmentation
    import numpy as np
    import mahotas as mh
    filt_cell = mh.gaussian_filter(cell, 2)
    T = mh.thresholding.otsu((np.rint(filt_cell).astype('uint8')))
    dist = mh.stretch(mh.distance(filt_cell > T))
    
    Bc = np.ones((3,3))
    rmax = mh.regmin((dist))
    rmax = np.invert(rmax)
    labels, num_cells = mh.label(rmax, Bc)
    surface = (dist.max() - dist)
    areas = mh.cwatershed(dist, labels)
    areas *= T
    return areas

num_frames = 4
num_chan = 2
max_w = 1300
max_h = 1000

def getImageData(inputImagePath):
    # set up image data, extract from tiff format
    # extract multiple channels (8 page tiff -> 4 images, with green and red channels)
    import sys
    if '/home/ccbr_okraus/cell_segmentation/Segmentation_library/' not in sys.path:
        sys.path.append('/home/ccbr_okraus/cell_segmentation/Segmentation_library/')

    import Load_GR
    from PIL import Image

    data_out = np.zeros((num_frames,num_chan,max_h,max_w))

    im = Image.open(curImagePath)
    green,red = Load_GR.load(im)
    green = Load_GR.convert(green)
    red = Load_GR.convert(red)

    for frame in range(num_frames):
        stacked_images = np.concatenate((red[frame][np.newaxis,:,:],green[frame][np.newaxis,:,:]),axis=0)
        mid = (green[0].shape[0]/2,green[0].shape[1]/2)
        data_out[frame,:,:,:] = stacked_images[:,mid[0] - max_h/2: mid[0] + max_h/2,
                                                       mid[1] - max_w/2: mid[1] + max_w/2] 
    return data_out


