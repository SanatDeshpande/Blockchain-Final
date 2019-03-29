def to_bits(msg):
    """
        Function returns bit-representation of message passed.
    """
    bit_list = list(map(bin, bytearray(msg, 'utf-8')))
    bits = ""
    for chunk in bit_list:
        for bit in chunk:
            if bit == "b":
                pass
            else:
                bits += bit

    return bits


def pad_msg(bits):
    """
        Function takes string of bits and returns the same bit sequence with
        zero-padding on the left to end up with a multiple of 512.
    """
    num_padding = len(bits) % 512
    padding = ""
    for i in range(512 - num_padding):
        padding += "0"

    return padding + bits


if __name__ == '__main__':
    """
        Can also change this to accept some command line args, if we want to hash a .txt file or just pass a message in
        through the command line.
    """
    message = to_bits("test string to be hashed here")
    if len(message) % 512 != 0:
        message = pad_msg(message)
    print(message)
    print(len(message))


