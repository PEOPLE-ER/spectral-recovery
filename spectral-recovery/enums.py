from enum import Enum 

class Indices(Enum):
    NDVI = "NDVI"
    NBR = "NBR"

    def __str__(self) -> str:
        return self.name
    
    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value == value.upper():
                return member