from base import Language
from c import CLanguage

__all__ = ['CUDALanguage']

class CUDALanguage(CLanguage):
    pass
    # TODO: optimisation of translate_statement_sequence, interleave read/write
    # accesses with computations
