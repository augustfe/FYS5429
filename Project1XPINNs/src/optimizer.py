

class GeneralOptimizer:
    """
    Wrapper class making optax and jaxopt optimizers have the same signature
    """

    def __init__(self, optimizer, is_optax=True) -> None:
        self.optimizer = optimizer
        self.is_optax = is_optax

    def init():
        pass

    def update():
        pass