import random
import time
from string import ascii_letters
import hashlib
from matplotlib import pyplot as plt
from Hash import DenseHash
from Hash import LSTMHash
import numpy as np


def GenerateMessage(multiple):
    """
    Generate random length message of a multiple of 512 bits
    :param multiple: How many multiples of 512 should be the message
    :return: Random message
    """
    data = np.zeros(512 * multiple)
    s = np.random.choice(len(data), np.random.randint(len(data)), replace=False)
    data[s] = 1
    return data


def test_speed(multiple, num_tests):
    """

    :param multiple: How long of a message it should be in a multiple of 512 bits
    :param num_tests: Number of tests to run as a comparison
    :return: Amount of time to run SHA256, MD5, LSTM, and Dense
    """
    test_str = ''.join(random.choice(ascii_letters) for j in range(512 * multiple)).encode('UTF-8')

    nn_test = GenerateMessage(multiple)

    # Testing SHA256 Run Time
    start = time.process_time()
    for i in range(0, num_tests):
        hashlib.sha256(test_str).hexdigest()
    stop = time.process_time()
    sha_time = (stop - start)


    # Testing MD5 Run Time
    start = time.process_time()
    for i in range(0,num_tests):
        hashlib.md5(test_str).hexdigest()
    stop = time.process_time()
    md5_time = (stop - start)

    # Testing LSTM
    model = LSTMHash()
    start = time.process_time()
    for i in range(0, num_tests):
        model.hash(nn_test).tostring()
    stop = time.process_time()
    LSTM_time = (stop - start)

    # Testing Dense Hash
    model = DenseHash()
    start = time.process_time()
    for i in range(0, num_tests):
        model.hash(nn_test).tostring()
    stop = time.process_time()
    Dense_time = (stop - start)

    return sha_time, md5_time, LSTM_time, Dense_time


def main():
    sha_times = []
    md5_times = []
    LSTM_times = []
    dense_times = []
    lengths = [i for i in range(1,11, 1)]
    for i in range(1, 11, 1):
        sha_time, md5_time, LSTM_time, Dense_time = test_speed(i, 1000)
        sha_times.append(sha_time)
        md5_times.append(md5_time)
        LSTM_times.append(LSTM_time)
        dense_times.append(Dense_time)


    plt.scatter(lengths, sha_times, color='b', label="SHA256")
    plt.scatter(lengths, md5_times, color='g', label="MD5")
    plt.scatter(lengths, LSTM_times, color='y', label="LSTM")
    plt.scatter(lengths, dense_times, color='r', label="Dense")
    plt.xlabel("Input string length")
    plt.ylabel("Average run time")
    plt.legend()
    plt.title("Running time by Input length (multiples of 512) for 1000 Hashes")
    plt.show()
    return 1


if __name__ == "__main__":
    main()
