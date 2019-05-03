from Hash import LSTMHash
from Hash import DenseHash
from Hash import DoubleDenseHash
import numpy as np
import random
import gc
import hashlib
from string import ascii_letters
from matplotlib import pyplot as plt

LOOPNUM = 50


def GenerateMessage():
    # Generate random length message between 1 * 512 and 20 * 512
    data = np.zeros(512 * random.randint(1, 21))
    s = np.random.choice(len(data), np.random.randint(len(data)), replace=False)
    data[s] = 1
    return data


def sha_subset_collision(bits):
    """Test for a collision with the number of bits for the sha256"""
    hash_number = []
    mask = 2**bits - 1
    for i in range(LOOPNUM):
        hash_number.append(sha_find_collision(mask))
    return hash_number


def sha_find_collision(mask):
    collision = False
    hashmap = dict()
    counter = 0
    while not collision:
        message = ''.join(random.choice(ascii_letters) for j in range(512 * random.randint(1, 21))).encode('UTF-8')
        outputHash = int(hashlib.sha256(message).hexdigest(), 16) & mask
        if outputHash in hashmap:
            collision = True
        hashmap[outputHash] = True
        counter = counter + 1
        gc.collect()
        if counter % 1000 == 0:
            print(counter)
    return counter


def subset_collision(hash_function, bits):
    """Test for a collision with the number of bits for the hash function"""
    hash_number = []
    for i in range(LOOPNUM):
        hash_number.append(find_collision(hash_function,bits))
        gc.collect()

    return hash_number


def find_collision(hash_function, bits):
    collision = False
    hashmap = dict()
    counter = 0
    model = hash_function()
    while not collision:
        message = GenerateMessage()
        outputHash = model.hash(message)[-bits:].tostring()
        if outputHash in hashmap:
            collision = True
        hashmap[outputHash] = True
        counter = counter + 1
        gc.collect()
        if counter % 1000 == 0:
            print(counter)
    return counter


def main():
    bits = 20
    sha = sha_subset_collision(bits)
    print(sum(sha)/LOOPNUM)
    print(max(sha))
    print(min(sha))
    #dense = subset_collision(DenseHash, bits)
    #lstm = subset_collision(LSTMHash, bits)
    #double = subset_collision(DoubleDenseHash, bits)

    #plt.boxplot([dense, lstm, double, sha], labels=['Dense', 'LSTM', 'DoubleDense', 'SHA256'])
    #plt.xlabel("Neural Network Hash function")
    #plt.ylabel("Hashes to collision")
    #plt.title("Collision Test on {} lowest bits".format(bits))
    #plt.show()

    return 0


if __name__ == "__main__":
    main()
