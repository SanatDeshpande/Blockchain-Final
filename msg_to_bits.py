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


def pad_msg(bits, length):
    """
        Function takes string of bits and returns the same bit sequence with
        pre-processed as in the SHA256 scheme.

    """

    bits = "1" + bits
    padding = ""

    while (len(padding) + len(bits)) % 512 != 448:
        padding += "0"

    padded_msg = format(length, "064b") + padding + bits

    return padded_msg



def bitify(msg):
    """
        Function puts together both pad_msg and to_bits functions for ease of calling.
    """
    length = len(msg)
    bits = to_bits(msg)

    if len(bits) % 512 != 0:
        bits = pad_msg(bits, length)

    return bits


if __name__ == '__main__':
    message = "test input message here"
    print(bitify(message))

    #print(len(bitify(message)))    # just to be sure it's a multiple of 512



