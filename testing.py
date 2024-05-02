#import encoder.encoder as enc
import numpy as np
import tensorflow as tf

from data.get_data import get_data
import encoder.encoder as enc
import decoder.decoder as dec
import losses.loss as loss

input_params = [
    "n_batch": 1, # batch size
    "patch_size": 12, # size of image patches
    "n_pca": 2, # size of the input to the network, PCA will be used to map to lower dimensions
    "overcomplete_ratio": 1.5, # this is the ratio of overcompletness of lateral dimensions. If bigger than zero, then lateral dimensions bigger then pixel dimensions
    "n_lat_samp":  1, # Number of samples to draw during training. Default can be 1
    "loss_type": "exp", # 
    "sigma" : np.float32(np.exp(-1.)), # std of noise


]


n_batch = input_params["n_batch"]
patch_size = input_params["patch_size"]
n_pca = input_params["n_pca"]
overcomplete_ratio = input_params["overcomplete_ratio"]
n_lat_samp = input_params["n_lat_samp"] 
loss_type = input_params["loss_type"] 

# reads the data, splitted into train + test split
data,varif,test,PCA = get_data(patch_size,n_pca,"BSDS",True)

# images = tf.Variable(tf.ones(shape=[n_batch,n_pca]), dtype=tf.float32)

images = tf.keras.layers.Input(batch_size =n_batch, shape=(n_pca))

n_lat = overcomplete_ratio * n_pca

lat_mean, lat_trans, lat_cor = enc.make_encoder(images,n_lat)

# Reparametrization: Sample from the latent space 
noise1 = tf.random.normal(shape = [int(lat_mean.shape[0]),n_lat_samp,1,int(lat_mean.shape[1])]) # Batch X Samples X 1 X lateral_dim
noise2 = tf.random.normal(shape = [int(lat_mean.shape[0]),n_lat_samp,1,int(lat_mean.shape[1])])

lv1 =  tf.reduce_sum(tf.expand_dims(lat_trans,1)*noise1,-1)
lv2 =  tf.reduce_sum(tf.expand_dims(lat_cor,1)*noise2,-1) #currently zero

noise = lv1 + lv2

# we sample from the distribution
latents_z = tf.expand_dims(lat_mean,1) + noise

reconstruction, weights = dec.make_decoder(latents,n_pca)

prior_params = {
    
}

loss_exp, recon_err = loss.compute_loss(loss_type,images,reconstruction,lat_mean,var,latents_z,sigma,prior_params)

asd = {
    "n_lat":n_lat,
    "images": images,

}