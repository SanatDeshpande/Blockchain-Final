import random
from string import ascii_letters
import hashlib
from matplotlib import pyplot as plt
import numpy as np
from Hash import LSTMHash
from Hash import DenseHash

def GenerateMessage(multiple):
    # Generate random length message between 512 and 26,112
    data = np.zeros(512 * multiple)
    s = np.random.choice(len(data), np.random.randint(len(data)), replace=False)
    data[s] = 1
    return data


def bit_variance_sha():
    num_set = np.zeros(256)
    for i in range(0,1024):
        test_str = ''.join(random.choice(ascii_letters) for j in range(256)).encode('UTF-8')
        digest = int(hashlib.sha256(test_str).hexdigest(), 16)
        for j in range(0, 256):
            num_set[j] += (digest & (1 << (255 - j))) >> (255-j)

    return num_set / 1024


def bit_variance_LSTM():
    num_set = np.zeros(256)
    model = LSTMHash()
    for i in range(0, 1024):
        test_str = GenerateMessage(1)
        digest = model.hash(test_str)
        for j in range(0, 256):
            num_set[j] += int(digest[j])

    return num_set / 1024


def bit_variance_Dense():
    num_set = np.zeros(256)
    model = DenseHash()
    for i in range(0, 1024):
        test_str = GenerateMessage(1)
        digest = model.hash(test_str)
        for j in range(0, 256):
            num_set[j] += digest[j]

    return num_set / 1024


def main():
    set_prob = bit_variance_sha()
    bit_num = [i for i in range(0,256)]

    plt.scatter(bit_num, set_prob)
    plt.xlabel("Bit Number")
    plt.ylabel("Percentage of time set")
    plt.title("SHA256 - Percentage of Bit Set")
    plt.ylim(0, 1)
    plt.figure()

    set_prob = bit_variance_LSTM()
    plt.scatter(bit_num, set_prob)
    plt.xlabel("Bit Number")
    plt.ylabel("Percentage of time set")
    plt.title("LSTM - Percentage of Bit Set")
    plt.ylim(0, 1)
    plt.figure()

    set_prob = bit_variance_Dense()
    plt.scatter(bit_num, set_prob)
    plt.xlabel("Bit Number")
    plt.ylabel("Percentage of time set")
    plt.title("Dense - Percentage of Bit Set")
    plt.ylim(0, 1)
    plt.show()

    return 0


if __name__ == "__main__":
    main()