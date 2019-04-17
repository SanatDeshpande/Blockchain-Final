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

    return list(map(int,list(bits)))


if __name__ == '__main__':
    message = "test input message here"
    #print(bitify(message))
    #print(pad_msg(''.join(str(x) for x in [0,0]),2))
    #print(pad_msg('00',2))
    zero_str = '00000000'
    print(zero_str[:0] + '1' + zero_str[1:1])
    #print(len(bitify(message)))    # just to be sure it's a multiple of 512



