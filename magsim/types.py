from enum import Enum


class SourceType(Enum):
    DIPOLE = "dipole"
    MONOPOLE = "monopole"
    QUADRUPOLE = "quadrupole"
    DF_SECS = "df_secs"
    CF_SECS = "cf_secs"


class NoiseModel(Enum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    MIXED = "mixed"
    PINK = "pink"
    CORRELATED = "correlated"
