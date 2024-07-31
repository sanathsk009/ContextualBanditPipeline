from .dataset_generator_base import DatasetGenerator
from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict, Sequence, Hashable
import numpy as np


class UcimlDatasetGenerator(DatasetGenerator):
    def __init__(self, normalize=True, uciml_id=59, max_arms=np.inf):
        # fetch dataset
        uciml_dataset = fetch_ucirepo(id=uciml_id)

        # data (as pandas dataframes)
        X = uciml_dataset.data.features.to_numpy()
        Y = uciml_dataset.data.targets.to_numpy().flatten()

        if normalize:
            scaler = MinMaxScaler()
            scaler.fit(X)
            X = scaler.transform(X)

        # Convert classification class to reward for correct guess.
        unique_elements = np.unique(Y)
        unique_len = min(max_arms, len(unique_elements))
        # Create a dictionary mapping unique elements to their indices
        element_to_index = {
            element: i % unique_len for i, element in enumerate(unique_elements)
        }

        # Define a function to create one-hot encoding for an element
        def encode_one_hot(element):
            return np.eye(unique_len)[element_to_index[element]]

        # Use map() to apply the encoding function to each element in the original list
        one_hot_encoded = list(map(encode_one_hot, Y))
        Y = np.array(one_hot_encoded)

        # store data
        self.X = X
        self.Y = Y
        self.name = uciml_dataset.metadata.name

    def generate_data(self, size=1000):
        # Schuffle data
        X, Y = self.X, self.Y
        p = np.random.randint(0, len(X), size)
        X, Y = X[p], Y[p]

        # Add noise.
        Y = (Y + 0.5) / 2
        Y = np.random.binomial(1, Y)

        return X, Y

    @property
    def params(self) -> Dict[str, Any]:
        return {"family": "UCIML" + self.name}
