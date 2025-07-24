from enum import Enum

# FRAGILITY MODEL params in milliseconds
WINSIZE_LTV = 250
STEPSIZE_LTV = 125
RADIUS = 1.5  # perturbation radius
PERTURBTYPE = "c"


class PERTURBATIONTYPES(Enum):
    COLUMN_PERTURBATION = "C"
    ROW_PERTURBATION = "R"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_
