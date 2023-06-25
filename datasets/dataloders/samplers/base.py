import abc

class BaseSample(object, metaclass=abc.ABCMeta):
    """
    # Base class for all sample
    """
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self):
        """
        # Generate heterogeneous indices of batches
        """
        pass

    @abc.abstractmethod
    def get_k(self)->int:
        """
        # Ensure batch_size % k == 0
        """
        pass
