import enum


class Frequency(enum.StrEnum):
    WEEKLY = "W-FRI"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"
