# Runs the hash function of your choice LOOPNUM times on random input until it finds a collision.
# Returns an average number of cycles until collision is found
# Use command line arg "dense" or "lstm" to change hash functions. dense by default

import sys
import numpy as np
import random
import gc
from Hash import DenseHash
from Hash import LSTMHash
from Hash import DoubleDenseHash

LOOPNUM = 4


def GenerateMessage():
    # Generate random length message between 1 * 512 and 20 * 512
    data = np.zeros(512 * random.randint(1, 21))
    s = np.random.choice(len(data), np.random.randint(len(data)), replace=False)
    data[s] = 1
    return data


def Hashloop(model):
    average = 0
    for i in range(LOOPNUM):
        # Hash until there is a collision
        isCollision = False
        counter = 0
        hashmap = dict()
        while not isCollision:
            message = GenerateMessage()
            outputHash = model.hash(message).tostring()
            if outputHash in hashmap:
                isCollision = True
            hashmap[outputHash] = True
            counter = counter + 1
            gc.collect()
            if counter % 1000 == 0:
                print(counter)
        average += counter
        print(counter)

    average = average / LOOPNUM
    return average


if __name__ == '__main__':
    model = DenseHash()
    if len(sys.argv) == 2:
        if sys.argv[1] == "lstm":
            model = LSTMHash()
        if sys.argv[1] == "double" or sys.argv[1] == "double_dense":
            model = DoubleDenseHash()

    average = Hashloop(model)
    print("There was a collision on the %sth hash" % average)
