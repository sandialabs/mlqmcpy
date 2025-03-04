def multilevel(func):
    """
    Decorator that attaches a multilevel variant of the solver as an attribute 'ml'
    on the original function.

    The multilevel variant computes:

        func(level, sample) - func(level - 1, sample)   for level > 0
        func(0, sample)                                 for level == 0

    """

    def ml_func(level, sample):
        qoi = func(level, sample)
        if level > 0:
            qoi -= func(level - 1, sample)
        return qoi

    func.ml = ml_func

    return func
