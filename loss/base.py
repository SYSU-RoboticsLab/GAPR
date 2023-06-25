import abc

class BaseLoss(object, metaclass=abc.ABCMeta):
    """
    # Base class for all loss functions

    """
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self):
        pass

    @abc.abstractmethod
    def print_stats(self):
        pass
