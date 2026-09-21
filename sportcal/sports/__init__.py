"""Sport registry. Importing this package registers every supported sport."""
from sportcal.sports.base import REGISTRY, Sport, get, names, register  # noqa: F401
from sportcal.sports import hockey, soccer  # noqa: F401  (registers hockey-nhl, hockey-iihf, soccer-fifa)
