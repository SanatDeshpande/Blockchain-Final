# Runs the hash function of your choice LOOPNUM times on random input until it finds a collision.
# Returns an average number of cycles until collision is found
# Use command line arg "dense" or "lstm" to change hash functions. dense by default

import sys
import numpy as np
import random
from Hash import DenseHash
from Hash import LSTMHash

LOOPNUM = 100


def GenerateMessage():
    # Generate random length message between 512 and 26,112
    data = np.zeros(512 * random.randint(1, 51))
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
            if counter % 100 == 0:
                print(counter)

    average = average / 100
    return average


if __name__ == '__main__':
    model = DenseHash()
    if len(sys.argv) == 2 and sys.argv[1] == "lstm":
        model = LSTMHash()

    average = Hashloop(model)
    print("There was a collision on the %sth hash" % average)
