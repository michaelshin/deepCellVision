import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from django.conf import settings

try:
    # faster implementation using bindings to libxml
    from lxml import etree as ET
except ImportError:
    print 'Falling back to default ElementTree implementation'
    from xml.etree import ElementTree as ET

import re
RE_NAME_LONG = re.compile('^(\d+)_Exp(\d+)Cam(\d+)$')
RE_NAME = re.compile('^Exp(\d+)Cam(\d+)$')

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
    plt.savefig(savepath, bbox_inches='tight', pad_inches = 0)
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")
    return savepath

def _segment(cell):
    # takes a numpy array of a microscopy
    # segments it based on filtering the image then applying a distance transform and
    # a watershed method to get the proper segmentation

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

def show_segment(path, name):
    from PIL import Image
    img = Image.open(path).convert('L')
    arr = np.asarray(img, np.uint8)
    arr = _segment(arr)
    plt.imshow(arr)
    loc = str(settings.MEDIA_ROOT + '/segment/' + name.split('.')[0])
    save(loc)
    np.save(loc, arr) #save the array to give as a raw file
    return

def _parse_xml(x):
    frames = cams = exps = arrays = 0
    for n in [e.get('Name') for e in ET.fromstring(x).findall('Arrays/Array[@Name]')]:
        # Print "n" if you want to see the values of the "Name" attribute
        # print n
        
        arrays += 1
        
        # Names found in Oren's flex files
        m = RE_NAME_LONG.match(n)
        if m:
            frames, exps, cams = [max(g) for g in zip(map(int, m.groups()), (frames, exps, cams))]
            continue
        
        # Names found in Mojca's flex files
        m = RE_NAME.match(n)
        if m:
            exps, cams = [max(g) for g in zip(map(int, m.groups()), (exps, cams))]
            frames = arrays / cams
            continue
        
        raise Exception('Unknown flex name pattern')

    return frames, exps, cams

def _r(fp):
    '''
        Read one byte as char and return byte value
    '''
    return ord(fp.read(1))

'''
    type reading utils
'''
def _get_short(fp):
    return _r(fp) + (_r(fp) << 8)

def _get_int(fp):
    return _r(fp) + (_r(fp) << 8) + (_r(fp) << 16) + (_r(fp) << 24)

def get_flex_data(im):
    im = open(im, 'rb')

    _mm = im.read(2)
    _ver = _get_short(im)
    _offs = _get_int(im)
    im.seek(_offs)
    
    _num_tags = _get_short(im)
    xml = None
    
    for _tag_idx in xrange(_num_tags):
        _tag = _get_short(im)
        _tag_type = _get_short(im)
        _tag_len = _get_int(im)
        if _tag_type == 3 and _tag_len == 1:
            _tag_value = _get_short(im)
            _ = _get_short(im)
        else:
            _tag_value = _get_int(im)
        
        if _tag == 65200:
            _saved_offs = im.tell()
            im.seek(_tag_value)
            xml = im.read(_tag_len)
            im.seek(_saved_offs)
    
    im.close()
    
    return _parse_xml(xml)

    
