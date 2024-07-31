# Implementing a base class for dataset creation.
from typing import Any, Dict, Sequence, Hashable


class DatasetGenerator:
    def __init__(self):
        pass

    def generate_data(self, size):
        pass

    @property
    def params(self) -> Dict[str, Any]:
        return {"family": "DatasetGenerator"}
