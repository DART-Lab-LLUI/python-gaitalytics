import decimal


def get_decimal_places(number: float) -> int:
    """Get the number of decimal places in a number.

    Args:
        number: The number to get the decimal places for.

    Returns:
        The number of decimal places in the number.
    """
    places = decimal.Decimal(str(number)).as_tuple().exponent
    if type(places) is not int:
        raise ValueError("The number of decimal places must be an integer.")

    return abs(places)
