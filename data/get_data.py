import numpy as np
from . import image_preprocess as IM
from sklearn.decomposition import PCA as PCA

DATASET_PATH = r"D:\DSUsers\uie72664\dataset\BSDS300\\"

def get_data(patch_size, nvar, dataset = "bruno", whiten = False,CNN = False):
    imlist = np.squeeze(IM.get_array_data(DATASET_PATH + "iids_train.txt"))
    data = imlist

    # Original whitening on the frequency domain
    '''ff0 = [np.expand_dims(np.fft.fftfreq(len(d)),1) for d in data]
    ff1 = [np.expand_dims(np.fft.fftfreq(len(d[0])),0) for d in data]
    ff = [np.sqrt(ff0[k]**2 + ff1[k]**2) for k in range(len(ff0))]
    f0 = 1./10

    mask = np.abs(ff) * np.exp(-(np.abs(ff)/f0)**4)

    data = [np.real(np.fft.ifft2(mask[k]*np.fft.fft2(data[k]))) for k in range(len(data))]'''

    data = [IM.get_filter_samples(DATASET_PATH + "images/train/" + i + ".jpg",size = patch_size) for i in imlist]

    data = np.concatenate(data)

    data = np.reshape(data,[-1,patch_size*patch_size])

    LL = len(data)
    var = data[:int(LL/10)]
    test = data[int(LL/10):int(2*LL/10)]
    data = data[int(2*LL/10):]

    print("BSDS data size: {}".format(data.shape))

    white = PCA(nvar,copy = True,whiten = False) ###THIS USED TO BE =whiten

    fit_data = data #white.fit_transform(data)
    fit_var = var #white.transform(var)
    fit_test = test #white.transform(test)

    fit_data = np.random.permutation(fit_data)
    fit_var = np.random.permutation(fit_var)
    fit_test = np.random.permutation(fit_test)

    # if CNN:
    #     fit_data = get_CNN_dat(fit_data,white,whiten)
    #     fit_var = get_CNN_dat(fit_var,white,whiten)
    #     fit_test = get_CNN_dat(fit_test,white,whiten)

    return np.float32(fit_data),np.float32(fit_var),np.float32(fit_test) ,white