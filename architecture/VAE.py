import tensorflow as tf
import numpy as np
import losses.loss as loss

class VAE(tf.keras.Model):
    def __init__(self, input_shape, n_pca, latent_size, loss_type, sigma):
        super(VAE, self).__init__()

        self.n_lateral_samples = 1

        # Loss params:
        self.loss_type =loss_type
        self.sigma = sigma

        # Encoder
        self.latent_dim = latent_size

        input_tensor = tf.keras.layers.InputLayer(input_shape=input_shape)
        # Shared hidden layer
        # TODO: input shape
        self.net = tf.keras.layers.Dense(128, activation = tf.nn.relu)
        # Mean hidden layers, not shared
        self.net11 = tf.keras.layers.Dense(256, activation = tf.nn.relu)
        self.net12 = tf.keras.layers.Dense(512, activation = tf.nn.relu)

        # Variance hidden layer, not shared
        self.net21 = tf.keras.layers.Dense(256,activation = tf.nn.relu)
        self.net22 = tf.keras.layers.Dense(512,activation = tf.nn.relu)
        
        # Mean recognition model
        self.mean = tf.keras.layers.Dense(latent_size)

         # Variance recognition model
        self.var = tf.keras.layers.Dense(latent_size, activation = lambda x: .01 + tf.nn.sigmoid(x))

        ## Latent mean and variance

        # Decoder
        nfeat = n_pca
        
        ish = latent_size
        self.decoder_weights = tf.Variable(name = "decoder_weights",initial_value = np.float32(np.random.normal(0,1./np.sqrt(int(ish)),[nfeat,int(ish)])))
        # self.decoder_weights = tf.keras.layers.Dense(int(nfeat), activation= 'sigmoid')
        # self.decoder_weights = tf.Variable(name = "decoder_weights",initial_value = np.float32(np.random.normal(0,1./np.sqrt(int(ish)),[nfeat,int(ish)])))
        
        

    def encode(self, x):
        x = self.net(x)
        x11 = self.net11(x)
        x12 = self.net12(x11)
        mean = self.mean(x12)

        x21 = self.net21(x)
        x22 = self.net22(x21)
        var = self.var(x22)

        # var = tf.map_fn(tf.linalg.diag, var, dtype=tf.float32)
        var_diag = tf.linalg.diag(var)

        #var2 = tf.zeros(tf.shape(var))
        return mean, var_diag, var
    
    def encode_partially(self, x):
        x = self.net(x)
        x11 = self.net11(x)
        return x11

    def decode(self, latent):
        return tf.reduce_sum(tf.expand_dims(latent,2) * tf.expand_dims(tf.expand_dims(self.decoder_weights,0),0),-1)
        # return self.decoder_weights(latent) 

    def reparametrization(self, lat_mean, lat_var, n_lat_samp):
        noise1 = tf.random.normal(shape = [int(lat_mean.shape[0]),n_lat_samp,1,int(lat_mean.shape[1])]) # Batch X Samples X 1 X lateral_dim

        lv1 =  tf.reduce_sum(tf.expand_dims(lat_var,1)*noise1,-1)

        noise = lv1

        latents_z = tf.expand_dims(lat_mean,1) + noise

        return latents_z

    def call(self, x):
        # lat_mean, lat_trans, lat_cor
        mean, var_diag, var = self.encode(x)

        latents_z = self.reparametrization(mean, var_diag, self.n_lateral_samples)

        x_hat = self.decode(latents_z)

        return x_hat, mean, var_diag, latents_z, var

    def compute_loss(self,batch,x_hat,lat_mean,var_diag,latents_z, var ):
        loss_type = self.loss_type
        sigma = self.sigma
        params = {
            "loss_type":loss_type
        }
        loss_exp, recon_err, recon_err_log, KL_error =loss.compute_loss(loss_type,batch,x_hat,lat_mean,var_diag,var,latents_z,sigma,params)
        return loss_exp, recon_err, recon_err_log, KL_error
    
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            x_hat, lat_mean, var_diag, latents_z, var = self.call(batch)  # Forward pass
            loss, rec_loss, recon_err_log, KL_error = self.compute_loss(batch,x_hat,lat_mean,var_diag,latents_z, var)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss+rec_loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss, rec_loss, recon_err_log, KL_error
    
