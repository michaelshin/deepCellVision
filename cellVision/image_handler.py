import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def RGB_to_gray(image):
    import numpy as np
    ''' 
    numpy array -> numpy array
    
    Returns a 2D greyscale image from a RGD representation of an image
    '''
    gray_image = np.zeros((image.shape[0], image.shape[1]))
    for row in range(len(image)):
        for col in range(len(image[row])):
            gray_image[row][col] = image[row][col][0]*0.299 + image[row][col][1]*0.587 + image[row][col][2]*0.114
    return gray_image


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


def show_segment(path):
    from PIL import Image
    from . import _segmentation
    from django.conf import settings
    import numpy as np
    import time
    img = Image.open(path)
    arr = np.asarray(img, np.uint8)
    if True:
        arr = RGB_to_gray(arr)
    arr = _segmentation._segment(arr)
    plt.imshow(arr)
    loc = settings.MEDIA_ROOT + time.strftime('/segment/%Y/%m/%d/array')
    save(loc)
    return

