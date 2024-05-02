import tensorflow as tf
import numpy as np

def generate_distribution(loss_type, params = {}):
    if loss_type == "gauss":
        return make_G()
    else:
        print("Issue with loss")
        exit()


def make_G(params):
    if "mean" not in params.keys():
        params["mean"] = np.float32(0)
    if "std" not in params.keys():
        params["std"] = np.float32(0)

    def f(latvals):
        return -tf.reduce_sum(((latvals - tf.expand_dims(params) )))
    
    return f,g,d