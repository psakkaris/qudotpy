
DIRAC_STR = "|%s>"

def dirac_str_to_int(dirac_str):
    """Converts a string in Dirac notation to an integer

    This will work for the computational basis only.
    Example: |1001> = 9

    Args:
        dirac_str: something like "|1001>"
    """
    return int(dirac_str[1:-1], 2)

def int_to_dirac_str(number, dimensionality=-1):
    """Convert an integer number to a bit string in Dirac Notation

    Can also be passed a dimensionality to pad the bit string with
    leading zeros. Example:
        2 with no dimensionality would be "|10>"
        2 with dimensionality 5 would be "|00010>"

    Args:
        number: the integer number to be converted
        dimensionality: if supplied, will be used to pad the bit string
                        with leading zeros

    Returns:
        bit string representation of the integer in Dirac Notation
    """
    bit_str = int_to_bit_str(number, dimensionality)
    return DIRAC_STR % bit_str

def int_to_bit_str(number, dimensionality=-1):
    """Convert an integer number to a bit string

    Can also be passed a dimensionality to pad the bit string with
    leading zeros. Example:
        2 with no dimensionality would be "10"
        2 with dimensionality 5 would be "00010"

    Args:
        number: the integer number to be converted
        dimensionality: if supplied, will be used to pad the bit string
                        with leading zeros

    Returns:
        bit string representation of the integer
    """
    bit_str = bin(number)[2:]
    if dimensionality > 0:
        missing_leading_zeros = dimensionality - len(bit_str)
        bit_str =  "0" * missing_leading_zeros + bit_str

    return bit_str

