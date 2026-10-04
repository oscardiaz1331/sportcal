"""Sport registry. Importing this package registers every supported sport."""
from sportcal.sports.base import REGISTRY, Sport, get, names, register  # noqa: F401
from sportcal.sports import basketball, hockey, soccer, tennis  # noqa: F401  (registers basketball-fiba, hockey-nhl, hockey-iihf, soccer-fifa, tennis-itf)
