class GeneralRewardParam:
    coeff: float = 0.
    targ: float = 0.

    def __init__(self, c: float = None, targ: float = None):
        self.coeff = c if c is not None else 0
        self.targ = targ if targ is not None else 0


class GaussianRewardParam(GeneralRewardParam):
    decay: float = 1.

    def __init__(self, c: float = None, d: float = None, targ: float = None):
        super().__init__(c, targ)
        self.decay = d if d is not None else 1


class CauchyRewardParam(GaussianRewardParam):
    order: int = 1

    def __init__(self, c: float = None, d: float = None, o: int = None, targ: float = None):
        super().__init__(c, d, targ)
        self.order = o if o is not None else 1


