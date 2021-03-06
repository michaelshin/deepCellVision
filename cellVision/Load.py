# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from PIL import Image
import numpy as np

# <codecell>

class ImageSequence:
    def __init__(self, im):
        self.im = im
    def __getitem__(self, ix):
        try:
            if ix:
                self.im.seek(ix)
            return self.im
        except EOFError:
            raise IndexError # end of sequence

# <codecell>

def load_GR(im):
    #im.seek(0)
    Red_in=[]
    Green_in=[]
    count=0
    for frame in ImageSequence(im):
        if count%2==0:
            Green_in.append(frame.copy())
        else:
            Red_in.append(frame.copy())
        count+=1
    return Green_in,Red_in

def load_Red_FarRed(im):
    #im.seek(0)
    im.seek(0)
    Green_in=im.copy()
    
    im.seek(1)
    Red_in=im.copy()
    
    im.seek(2)
    FarRed_in=im.copy()

    return [Green_in],[Red_in],[FarRed_in]

# <codecell>

def convert(Red):
    #convert to nparray object
    npArrayRed=[]
    for RedImage in Red:
        npArrayRed.append(np.array(RedImage.getdata()).reshape(RedImage.size[::-1]))
    return npArrayRed

