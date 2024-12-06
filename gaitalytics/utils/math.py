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


def find_local_minimas(arr):
    """
    Finds the indices of local minima in an array. A point is considered a local minima
    if it is smaller than the two points before and after it.

    Args:
        arr: The array to find minimas in

    Returns:
        A list of indices where the local minima are located.
    """
    local_minimas = []
    for i in range(2, len(arr) - 2):
        if (
            arr[i] < arr[i - 2]
            and arr[i] < arr[i - 1]
            and arr[i] < arr[i + 1]
            and arr[i] < arr[i + 2]
        ):
            local_minimas.append(i)

    return local_minimas
