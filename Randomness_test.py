from string import ascii_letters
import hashlib
import random
from matplotlib import pyplot as plt
from Hash import DenseHash
from Hash import LSTMHash
import numpy as np


def GenerateMessage(multiple):
    """
        Generate random length message between 512 and 26,112
    """
    data = np.zeros(512 * multiple)
    s = np.random.choice(len(data), np.random.randint(len(data)), replace=False)
    data[s] = 1
    return data


def randomness_test_control(num_test):
    """
    Control Test for Randomness Test using SHA256. Takes a message, flips a bit, and compares the two hashes.
    Half of the bits should be different between the two messages.
    :param num_test: Number of tests to be run
    :return: The average number of bits that are flipped for SH
    """
    sha_num_set = 0
    for i in range(0, num_test):
        num_set = np.zeros(256)
        test_str_one = random.getrandbits(512)
        flip = np.random.randint(0,512)
        test_str_two = test_str_one ^ (1 << flip)

        digest_one = int(hashlib.sha256(str(test_str_one).encode('UTF-8')).hexdigest(), 16)
        digest_two = int(hashlib.sha256(str(test_str_two).encode('UTF-8')).hexdigest(), 16)

        digest_xor = digest_one ^ digest_two

        for j in range(0, 256):
            sha_num_set += (digest_xor & (1 << (255 - j))) >> (255-j)
    return sha_num_set/num_test


def randomness_test_nn(num_test):
    """
    Control Test for Randomness Test using SHA256. Takes a message, flips a bit, and compares the two hashes.
    Half of the bits should be different between the two messages.
    :param num_test: Number of tests to be run
    :return: The average number of bits that are flipped for dense and LSTM
    """
    dnum_set = 0
    lnum_set = 0
    for i in range(0, num_test):
        msg_one = GenerateMessage(1)
        flip = np.random.randint(0, 512)
        msg_two = np.copy(msg_one)
        msg_two[flip] = (msg_two[flip] + 1) % 2
        dense_model = DenseHash()
        lstm_model = LSTMHash()
        dhash_one = dense_model.hash(msg_one)
        dhash_two = dense_model.hash(msg_two)

        lhash_one = lstm_model.hash(msg_one)
        lhash_two = lstm_model.hash(msg_two)

        for j in range(0,256):
            dnum_set += dhash_one[j] != dhash_two[j]
            lnum_set += lhash_one[j] != lhash_two[j]

    return dnum_set / num_test, lnum_set / num_test


def main():
    dense_avg, lstm_avg = randomness_test_nn(100)
    sha_avg = randomness_test_control(100)
    print(dense_avg)
    print(lstm_avg)
    print(sha_avg)
    return 0


if __name__ == "__main__":
    main()
