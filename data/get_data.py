import numpy as np
from . import image_preprocess as IM
from sklearn.decomposition import PCA as PCA
import os
import numpy as np
from PIL import Image
import pickle


DATASET_PATH = r"C:\Users\uie72664\Documents\data\\"

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

def getMNIST():
    imlist = []
    data = []
    total_path = r'C:\Users\uie72664\Documents\data2\MNIST\training'
    digit_numbers = os.listdir(total_path)
    for digit_number in digit_numbers:
        d_inames = total_path + '/' + digit_number
        im_names = os.listdir(d_inames)
        for idx, im_name in enumerate(im_names):
            total_im_name = d_inames + '/' + im_name
            imlist.append(total_im_name)
            img = Image.open(total_im_name)

            PX = img.size
            imsize = PX[1]
            newsize = (PX[0]*imsize/PX[1],PX[1]*imsize/PX[1])
            img = img.resize(newsize)
            img = np.array(img)
            img = np.reshape(img,(img.shape[0],-1))

            data.append(img)

            if idx == 1000:
                break
  
    data_2 = np.array(data)
    data_2 = np.reshape(data_2,[-1,28*28])

    LL = len(data_2)
    var = data_2[:int(LL/10)]
    test = data_2[int(LL/10):int(2*LL/10)]
    data = data_2[int(2*LL/10):]

    fit_data = np.random.permutation(data)
    fit_var = np.random.permutation(var)
    fit_test = np.random.permutation(test)

    return np.float32(fit_data),np.float32(fit_var),np.float32(fit_test) ,None

def getTexture():
    file = open(r'C:\Users\uie72664\Documents\data2\Texture\fakelabeled_natural_commonfiltered_640000_20px.pkl', 'rb')
    data = pickle.load(file)
    file.close()
    return data['train_images'],data['test_images'],data['test_images'] ,None

def get_data_original(patch_size, nvar, dataset = "bruno", whiten = False,CNN = False):
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