import tensorflow as tf


def make_encoder(input_tensor, latent_size):
    '''
    Creates the encoder
    param: 
    return: ouput of net1: mean
    return: output of net2: variance
    return:
    '''


    # Shared hidden layer
    net = tf.keras.layers.Dense(128, activation = tf.nn.relu)(input_tensor)

    # Mean hidden layers, not shared
    net1 = tf.keras.layers.Dense(256, activation = tf.nn.relu)(net)
    net1 = tf.keras.layers.Dense(512, activation = tf.nn.relu)(net1)

    # Variance hidden layer, not shared
    net21 = tf.keras.layers.Dense(256,activation = tf.nn.relu)(net)
    net2 = tf.keras.layers.Dense(512,activation = tf.nn.relu)(net21)

    # Mean recognition model
    mean = tf.keras.layers.Dense(latent_size)(net1)

    # Variance recognition model
    var = tf.keras.layers.Dense(latent_size, activation = lambda x: .01 + tf.nn.sigmoid(x))(net2)
    # var = tf.map_fn(tf.linalg.diag, var, dtype=tf.float32)
    var = tf.linalg.diag(var)

    var2 = tf.zeros(tf.shape(var))

    return mean, var, var2