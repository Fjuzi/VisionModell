import numpy as np
import pickle
import os
import time
import glob
import json
import tensorflow as tf

def trainable(scope = "",match = True):
    a = tf.trainable_variables()
    if match:
        return [x for x in a if x.name[:len(scope)] == scope]
    else:
        return [x for x in a if x.name[:len(scope)] != scope]

def save_dict(direc,dic):
    F = open(direc + ".json","w")
    json.dump(dic,F)
    F.close()

def read_dict(direc):
    F = open(direc + ".json","r")
    out = json.load(F)
    F.close()

    return out

def get_directory(direc="./",tag = "results"):
    while os.path.exists(direc + "lockfile.lock"):
        time.sleep(1)

    F = open(direc + "lockfile.lock", 'w')
    F.close()

    directories = os.walk(direc)
    directories = [x for d in directories for x in d[1] if d[0] == direc if x[:len(tag)] == tag]

    if len(directories) == 0:
        fnum = 0
    else:
        fnum = [int(x.split(tag)[1][1:].split(".")[0]) for x in directories]
        fnum.sort()
        fnum = fnum[-1]+1

    newdir = direc + tag + "_" + str(fnum)

    os.mkdir(newdir)

    os.remove(direc + "lockfile.lock")

    return newdir
        
def dump_file(name,data):
    F = open(name,"wb")
    pickle.dump(data,F)
    F.close()
    
def fetch_file(name):
    F = open(name,"rb")
    out = pickle.load(F)
    F.close()
    return out

def csv_line(value_parser):
    """
    Return a function that parses a line of comma separated values.

    Examples:

    >>> csv_line(int)('1, 2, 3')
    [1, 2, 3]
    >>> csv_line(float)('0.5, 1.5')
    [0.5, 1.5]

    For example, it can be passed to type argument of
    `argparse.ArgumentParser.add_argument` as follows::

        parser.add_argument(
            ...,
            type=csv_line(float),
            help='Comma separated value of floats')

    """
    def convert(string):
        return list(map(value_parser, string.split(',')))
    return convert

def read_csv(f,header = False):
    F = open(f,"r")
    out = []
    ll = 0
    for l in F:
        temp = l.split(",")
        temp[-1] = temp[-1][:-1]
        if (header and ll == 0):
            out.append(temp)
        else:
            out.append([float(x) for x in temp])
            
        ll += 1
        
    return out