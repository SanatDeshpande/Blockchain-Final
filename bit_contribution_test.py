import numpy as np
import msg_to_bits
from Hash import DenseHash


def bit_contribution_test(hash_function):
    """
    Creates strings of length 0 to 2048 bits, each with a single bit set and then checks for collisions.
    :param hash_function: NN to test
    :return: Number of  collisions
    """

    model = hash_function()
    hash_list = []
    zero_str = '0' * 2048
    for i in range(1, 2049):
        for j in range(0, i):
            flip_str = zero_str[:j] + '1' + zero_str[j+1:i]
            hash_list.append(list(map(int, list(msg_to_bits.pad_msg(flip_str, i)))))
        if i % 200 == 0:
            print(i)

    hashed_dict = dict()
    collisions = 0
    i = 0
    for to_hash in hash_list:
        i += 1
        hash_val = model.hash(to_hash, False).tostring()
        if hash_val in hashed_dict:
            collisions += 1
        hashed_dict[hash_val] = True
        if i % 10000 == 0:
            print(i)

    return collisions


def main():
    bit_contribution_test(DenseHash)
    return 0


if __name__ == "__main__":
    main()
