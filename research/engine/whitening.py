from research.engine.operator import SpectralOperator, SpectralOperatorFactory
from research.engine.basis import SpectralBasis

class WhitenOperator:
    @staticmethod
    def create(basis: SpectralBasis) -> SpectralOperator:
        return SpectralOperatorFactory.createWhiten(basis)

class UnwhitenOperator:
    @staticmethod
    def create(basis: SpectralBasis) -> SpectralOperator:
        return SpectralOperatorFactory.createUnwhiten(basis)
