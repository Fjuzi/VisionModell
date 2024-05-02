import numpy as np
from PIL import Image

def get_array_data(f,convert_to_float = False):
    F = open(f,"r")
    out = []

    for l in F:
        temp = l.split(",")
        temp[-1] = temp[-1][:-1]
        if convert_to_float:
            out.append([float(x) for x in temp])
        else:
            out.append([str(x) for x in temp])
            
    return np.array(out)

def split_by_size(A,s):
    SH = A.shape

    return np.array([A[i:i+s,j:j+s] for i in range(0,SH[0]-s,s) for j in range(0,SH[1]-s,s)])

def get_filter_samples(iname,size = 15,imsize = "def"):
    '''
    parameters:
      iname : the path to the image file to process
      nf    : number of "surround" filter patches
      na    : number of angles
      k     : frequency of gabor
      s     : scale (std.) of gabor
      t     : total size of gabor filter to compute (should be >> s)
      d     : distance between filters (also changes sampling rate)
      imsize: a standard size to reshape the 2nd axis to (while preserving the aspect ratio). 
   
    '''
    img = Image.open(iname)
    PX = img.size

    if imsize == "def":
        imsize = PX[1]
    
    newsize = (PX[0]*imsize/PX[1],PX[1]*imsize/PX[1])
    img = img.resize(newsize)

    img = np.array(img).mean(axis = 2)/255.#conv. to grayscale

    OT = split_by_size(img,size)

    return np.reshape(OT,(OT.shape[0],-1))