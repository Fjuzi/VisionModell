#import encoder.encoder as enc
import numpy as np
import tensorflow as tf
import logger.log as log
import time

from data.get_data import get_data
import encoder.encoder as enc
import decoder.decoder as dec
import losses.loss as loss
import utils.variance as make_var
import utils.utilities as util



input_params = { 
        "n_batch": 1, # batch size
        "patch_size": 12, # size of image patches
        "n_pca": 2, # size of the input to the network, PCA will be used to map to lower dimensions
        "overcomplete_ratio": 1.5, # this is the ratio of overcompletness of lateral dimensions. If bigger than zero, then lateral dimensions bigger then pixel dimensions
        "n_lat_samp":  1, # Number of samples to draw during training. Default can be 1
        "loss_type": "exp", # 
        "sigma" : np.float32(np.exp(-1.)), # std of noise
    
}




n_batch = input_params["n_batch"]
patch_size = input_params["patch_size"]
n_pca = input_params["n_pca"]
overcomplete_ratio = input_params["overcomplete_ratio"]
n_lat_samp = input_params["n_lat_samp"] 
loss_type = input_params["loss_type"] 
sigma = input_params["sigma"]

dirname = util.get_directory(direc="./model_output/",tag = loss_type)
np.random.seed(110)

LOG = log.log(dirname + "/logfile.csv")




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

var,DC,CC = make_var.get_var_mat(lat_trans,lat_cor)

reconstruction, weights = dec.make_decoder(latents_z,n_pca)

prior_params = {
    "loss_type":loss_type,
}

loss_exp, recon_err = loss.compute_loss(loss_type,images,reconstruction,lat_mean,var,latents_z,sigma,prior_params)

netpar = {
    "n_lat":n_lat,
    "images": images,
    "variance": var,
    "diagonal_var":DC,
    "factor_var":CC,
    "reconstruction":reconstruction,
    "weights":weights,
    "loss_exp":loss_exp,
    "recon_err":recon_err,
    "latents":latents_z,
    "noise":noise,
    "mean":lat_mean,
    "diag_trans":lat_trans,
    "diag_cor":lat_cor,
    "data":data,
    "vardat":varif,
    "testdat":test,
    "PCA":PCA
}

# -------------------------------- This is the main part ----------------------


var = netpar["variance"]
loss_exp = netpar["loss_exp"]
recon_err = netpar["recon_err"]
images = netpar["images"]
data = netpar["data"]
varif = netpar["vardat"]

final_learning_rate = 0.00001

learning_rate = 0.001

n_grad_step = 1000000

LR_factor = np.float32(np.exp(-np.log(learning_rate/final_learning_rate)/n_grad_step))

LR= tf.Variable(np.float32(learning_rate),trainable = False)

adam  = tf.keras.optimizers.Adam(learning_rate = LR)
train = adam.minimize(loss_exp)

update_LR = tf.assign(LR,LR*LR_factor)

# --------------------------- RUN  TRAINING LOOP ------------------------------

# run_training_loop(data,varif,images,netpar["mean"],n_batch,train,loss_exp,recon_err,LOG,dirname,log_freq,n_grad_step,param_save_freq,update_LR)


data = data
vdata = varif
input_tensor = images
pos_mean = netpar["mean"]
batch_size = n_batch
# train_op,
loss_op = loss_exp
recerr_op = recon_err
log = LOG
dirname = dirname
log_freq = 1
n_grad_step = n_grad_step
save_freq = 10
update_LR = update_LR

def split_by_batches(data,batch_size,shuffle = True):
    if shuffle:
        D = data[np.random.permutation(range(data.shape[0]))]
    else:
        D = data

    D = np.array([D[k:k + batch_size] for k in range(0,len(data)-batch_size,batch_size)])
    return D

def var_loss(session,vdat,nbatch = 10):
    D = split_by_batches(vdat,batch_size,shuffle = False)
    loss = 0
    rerr = 0
    nb = 0
    means = []
    for d in D:
        nb += 1
        l,r,m = session.run([loss_op,recerr_op,pos_mean],{input_tensor:d})
        loss += l
        rerr += r
        means.append(m)
        if nb == nbatch:
            break
    loss /= nbatch
    rerr /= nbatch

    return loss,rerr,np.concatenate(means,axis = 0)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

nloss = 0
t1 = time.time()
av_time = -1
efrac = .9

log.log(["grad_step","loss","recloss","var_loss","var_rec","learning_rate","time_rem"],PRINT = True)

t_loss_temp = []
t_rec_temp = []

lrflag = True

saver = tf.train.Saver(max_to_keep = 1000)

@tf.function
def train_step(batch, labels):
    with tf.GradientTape() as tape:
        predictions = model(batch, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)


for grad_step in range(n_grad_step + 1):
    batch = data[np.random.choice(np.arange(len(data)),batch_size)]

    _,loss,recloss,newLR = sess.run([train_op,loss_op,recerr_op,update_LR],{input_tensor:batch}) # Run a session to get the loss/reconstruction error

    t_loss_temp.append(loss)
    t_rec_temp.append(recloss)

    if grad_step % log_freq  == 0:
        if grad_step == 0:
            av_time = -1
        elif grad_step != 0 and  av_time < 0:
            av_time = (time.time() - t1)
        else:
            av_time = efrac*av_time + (1. - efrac)*(time.time() - t1)