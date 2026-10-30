"""Independent torch reference for element-wise add."""


def elementwise_add(x, y):
    """Return x + y through the regular torch dispatch path.

    This is the accuracy reference used by the tests: it deliberately does not
    call the Ascend C operator, so a bug in the kernel cannot hide behind the
    same bug in the reference.
    """
    return x + y
