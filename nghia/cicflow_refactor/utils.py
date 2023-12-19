import uuid
from itertools import islice, zip_longest
from decimal import Decimal

import numpy

from loguru import logger


def grouper(iterable, n, max_groups=0, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""

    if max_groups > 0:
        iterable = islice(iterable, max_groups * n)

    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def random_string():
    return uuid.uuid4().hex[:6].upper().replace("0", "X").replace("O", "Y")


def get_statistics(alist: list):
    """Get summary statistics of a list"""
    iat = dict()

    if len(alist) > 1:

        alist = [Decimal(x) for x in alist]

        iat["total"] = sum(alist) # Fix this thing rerutn Decimal only
        iat["max"] = max(alist)
        iat["min"] = min(alist)
        # iat["mean"] = numpy.mean(alist)

        """ChatGPT Fix"""
        sum_alist = sum(alist)

        # If sum_alist is not a Decimal, convert it
        if not isinstance(sum_alist, Decimal):
            sum_alist = Decimal(str(sum_alist))

        mean_decimal = sum_alist / Decimal(len(alist))
        iat["mean"] = mean_decimal

        # Old Code
        # iat["std"] = numpy.sqrt(numpy.var(alist))

        # Calculate variance using Decimal
        variance_decimal = sum((x - mean_decimal) ** 2 for x in alist) / Decimal(len(alist))
        std_decimal = variance_decimal.sqrt()
        iat["std"] = std_decimal

    else:
        iat["total"] = 0
        iat["max"] = 0
        iat["min"] = 0
        iat["mean"] = 0
        iat["std"] = 0

    return iat
