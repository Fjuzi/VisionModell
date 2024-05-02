import numpy as np
import tensorflow as tf

import utils.distributions as dist

PI = np.float32(np.pi)
two = np.float32(2)


def get_log_recon_err(image, recon, sig):
    coef = tf.log(np.float32(2)*PI*(sig*sig))/np.float32(2)

    lat = tf.pow((tf.expand_dims(image,1) - recon)/sig,2)

    mean = tf.reduce_mean(coef + lat,axis = 1)

    return tf.reduce_sum(mean,axis = 1)

def get_log_gauss_D(mean,var,latent):
    '''
    
    '''


def KL_loss(a,b):
    '''
    Calculates the difference between between the elements
    We determine how many elements we sample from both a and b distribution
    And determine how much they differ
    (If they are the same distribution it should differ very little)
    param a: [batch_size X sample]
    param b: [batch_size X sample]
    '''
    return tf.reduce_mean(a-b, 1)

def compute_loss(loss_type,image,recon,mean,var,latvals,sig,params):
    ''' 
    Compute the loss of the network
    :param loss_type:
    :param image: original input image [B x Imsize]
    :param recon: reconstructed image [B x Imsize]
    :param mean: lateral representation mean
    :param var: lateral repesentation variance
    :param latvals: this is z: the value sampled from the latent space [batch_size X sample]
    :param sig:
    :param params: params for the loss_type (to the given distribution)
    :return loss_exp: expectation error
    :return recon_err: reconstruction error
    '''
    # Calculate E step:
    recon_err = get_log_recon_err(image,recon,sig)

    posterior = get_log_gauss_D(mean,var,latvals)

    # We calculate it, bc we want to mimic as if it were Gaussian
    f,g,d = dist.get_distribution(loss_type,params)
    prior = f(latvals)

    # Lets match the posterior to the prior (?)
    # in order to force the posterior to be Gaussian-like
    # this forced the prior to be SPARSE
    KL_error = KL_loss(posterior, prior)

    # Calculate M step:
    reconstruction_error = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(tf.pow(tf.expand_dims(image,1) - recon,2),axis = 1),axis = 1))

    return KL_error, reconstruction_error



