import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def save(path, ext='png', close=True, verbose=False):
    import os
    """Save a figure from pyplot.

    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.

    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.

    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.

    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.

    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")
    return savepath

def show_segment(path, name):
    from PIL import Image
    from . import _segmentation
    from django.conf import settings
    import numpy as np
    img = Image.open(path).convert('L')
    arr = np.asarray(img, np.uint8)
    arr = _segmentation._segment(arr)
    plt.imshow(arr)
    loc = str(settings.MEDIA_ROOT + '/segment/' + name.split('.')[0])
    save(loc)
    np.save(loc, arr) #save the array to give as a raw file
    return

'''def classify(path):
    curImages = getImageData(curImagePath)
    curImages = normalize_by_constant_values(curImages,norm_vals['means'],norm_vals['stdevs'])
    break
    pred = model.get_test_time_crops({'X0':gnp.garray(curImages)})
    prediction_dict[orf]=np.mean(pred,axis=0)
    nn.ForwardProp({'X0':gnp.garray(curImages)})
    plt.figure(figsize=(12,12))
    for i in range(len(localizationTerms)):
        plt.subplot(4,5,i+1)
        plt.imshow(nn._layers['MIL_pool'].Z[image][i].as_numpy_array(),'gray',vmax=1,vmin=0,interpolation='none')
        plt.axis('off')
        plt.title(localizationTerms[i])
    plt.tight_layout()

    pred_maps = nn._layers['MIL_pool'].Z[image].as_numpy_array()
    area = pred_maps.sum(1).sum(1) / pred_maps.sum()
    area2 = pred_maps.sum(1).sum(1) / (pred_maps.shape[1]*pred_maps.shape[2])
'''
