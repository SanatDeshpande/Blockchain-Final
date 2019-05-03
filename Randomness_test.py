import hashlib
import random
from Hash import DenseHash
from Hash import LSTMHash
from Hash import DoubleDenseHash
import numpy as np
from matplotlib import pyplot as plt


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
    sha_num_set_list = []
    for i in range(0, num_test):
        sha_num_set = 0
        test_str_one = random.getrandbits(512)
        flip = np.random.randint(0, 512)
        test_str_two = test_str_one ^ (1 << flip)

        digest_one = int(hashlib.sha256(str(test_str_one).encode('UTF-8')).hexdigest(), 16)
        digest_two = int(hashlib.sha256(str(test_str_two).encode('UTF-8')).hexdigest(), 16)

        digest_xor = digest_one ^ digest_two

        for j in range(0, 256):
            sha_num_set += (digest_xor & (1 << (255 - j))) >> (255 - j)
        sha_num_set_list.append(sha_num_set)
    return sha_num_set_list


def randomness_test_nn(num_test):
    """
    Control Test for Randomness Test using SHA256. Takes a message, flips a bit, and compares the two hashes.
    Half of the bits should be different between the two messages.
    :param num_test: Number of tests to be run
    :return: The average number of bits that are flipped for dense and LSTM
    """
    dnum_set = 0
    ddnum_set = 0
    lnum_set = 0
    for i in range(0, num_test):
        msg_one = GenerateMessage(1)
        flip = np.random.randint(0, 512)
        msg_two = np.copy(msg_one)
        msg_two[flip] = (msg_two[flip] + 1) % 2
        dense_model = DenseHash()
        double_model = DoubleDenseHash()
        lstm_model = LSTMHash()
        dhash_one = dense_model.hash(msg_one)
        dhash_two = dense_model.hash(msg_two)

        ddhash_one = double_model.hash(msg_one)
        ddhash_two = double_model.hash(msg_two)

        lhash_one = lstm_model.hash(msg_one)
        lhash_two = lstm_model.hash(msg_two)

        for j in range(0, 256):
            dnum_set += dhash_one[j] != dhash_two[j]
            ddnum_set += ddhash_one[j] != ddhash_two[j]
            lnum_set += lhash_one[j] != lhash_two[j]

    return dnum_set / num_test, lnum_set / num_test, ddnum_set / num_test


def randomness_test(hash_function, num_test):
    """
    Generic Randomness Test. Takes a message, flips a bit, and compares the two hashes.
    Half of the bits should be different between the two messages.
    :param hash_function: Function under test
    :param num_test: Number of tests
    :return: Array of number of bits that are flipped per test
    """
    num_set_list = []
    for i in range(0, num_test):
        num_set = 0
        model = hash_function()
        msg_one = GenerateMessage(1)
        flip = np.random.randint(0, 512)
        msg_two = np.copy(msg_one)
        msg_two[flip] = (msg_two[flip] + 1) % 2
        hash_one = model.hash(msg_one)
        hash_two = model.hash(msg_two)
        for j in range(0, 256):
            num_set += hash_one[j] != hash_two[j]
        num_set_list.append(num_set)

    return num_set_list


def main():

    dense = randomness_test(DenseHash, 100)
    lstm = randomness_test(LSTMHash, 100)
    double = randomness_test(DoubleDenseHash, 100)
    sha = randomness_test_control(100)

    plt.boxplot([dense, lstm, double, sha], labels=['Dense', 'LSTM', 'DoubleDense', 'SHA256'])
    plt.xlabel("Neural Network Hash function")
    plt.ylabel("Bits Flipped")
    plt.title("Randomness Test for Hash Functions")
    plt.show()

    return 0


if __name__ == "__main__":
    main()
