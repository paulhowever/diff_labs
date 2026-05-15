from dataclasses import dataclass


@dataclass(frozen=True)
class VariantConfig:
    number: int
    omega0: float
    delta: float
    x0: float
    v0: float
    estimated: str
    T: float


VARIANT_6 = VariantConfig(
    number=6,
    omega0=2.0,
    delta=0.2,
    x0=1.0,
    v0=0.0,
    estimated="delta",
    T=15.0,
)


VARIANT = VARIANT_6


NOISE_STD = 0.05
SEED = 42
