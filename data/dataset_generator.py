import numpy as np
import pandas as pd
from models.fuzzy_model import FuzzyModel

class DatasetGenerator:
    def __init__(self, data_count: int, random_seed: int = 42):
        self.data_count = data_count
        self.random_seed = random_seed
        self.delivery_delay = None
        self.quality_defects = None
        self.completeness = None
        self.price_deviation = None
        self.payment_delay = None
        self.reliability = None

    def generate_dataset(self) -> pd.DataFrame:
        np.random.seed(self.random_seed)

        self.delivery_delay = np.random.randint(0, 31, size=self.data_count)
        self.quality_defects = np.random.randint(0, 21, size=self.data_count)
        self.completeness = np.random.randint(70, 101, size=self.data_count)
        self.price_deviation = np.random.randint(-15, 16, size=self.data_count)
        self.payment_delay = np.random.randint(0, 61, size=self.data_count)

        df = pd.DataFrame({
            'delivery_delay': self.delivery_delay,
            'quality_defects': self.quality_defects,
            'completeness': self.completeness,
            'price_deviation': self.price_deviation,
            'payment_delay': self.payment_delay
        })

        return df







