import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

class FuzzyModel:
    def __init__(self, pd_df: pd.DataFrame):
        self.df = pd_df
        self.delivery_delay = None
        self.quality_defects = None
        self.completeness = None
        self.price_deviation = None
        self.payment_delay = None
        self.reliability = None
        self.reliability_ctrl = None
        self.simulation = None

    def setup_fuzzy_variables(self):
        self.delivery_delay = ctrl.Antecedent(np.arange(0, 31, 1), 'delivery_delay')
        self.quality_defects = ctrl.Antecedent(np.arange(0, 21, 1), 'quality_defects')
        self.completeness = ctrl.Antecedent(np.arange(70, 101, 1), 'completeness')
        self.price_deviation = ctrl.Antecedent(np.arange(-15, 16, 1), 'price_deviation')
        self.payment_delay = ctrl.Antecedent(np.arange(0, 61, 1), 'payment_delay')

        self.reliability = ctrl.Consequent(np.arange(0, 101, 1), 'reliability')

    def setup_membership(self):
        # Средняя задержка поставки
        self.delivery_delay['низкая'] = fuzz.trimf(self.delivery_delay.universe, [0, 0, 3, 7])
        self.delivery_delay['средняя'] = fuzz.trimf(self.delivery_delay.universe, [5, 12, 18])
        self.delivery_delay['высокая'] = fuzz.trimf(self.delivery_delay.universe, [15, 22, 30, 30])

        # Доля дефектной продукции
        self.quality_defects['низкая'] = fuzz.trimf(self.quality_defects.universe, [0, 0, 2, 5])
        self.quality_defects['средняя'] = fuzz.trimf(self.quality_defects.universe, [4, 9, 14])
        self.quality_defects['высокая'] = fuzz.trimf(self.quality_defects.universe, [12, 16, 20, 20])

        # Доля поставки, полученная полностью
        self.completeness['низкая'] = fuzz.trimf(self.completeness.universe, [70, 70, 78, 85])
        self.completeness['средняя'] = fuzz.trimf(self.completeness.universe, [82, 89, 95])
        self.completeness['высокая'] = fuzz.trimf(self.completeness.universe, [92, 97, 100, 100])

        # Отклонение цены от согласованной
        self.price_deviation['выгодная'] = fuzz.trimf(self.price_deviation.universe, [-15, -15, -8, -2])
        self.price_deviation['нормальная'] = fuzz.trimf(self.price_deviation.universe, [-4, 0, 4])
        self.price_deviation['невыгодная'] = fuzz.trimf(self.price_deviation.universe, [2, 8, 15, 15])

        # Просрочка оплаты
        self.payment_delay['низкая'] = fuzz.trimf(self.payment_delay.universe, [0, 0, 10, 20])
        self.payment_delay['средняя'] = fuzz.trimf(self.payment_delay.universe, [15, 30, 45])
        self.payment_delay['высокая'] = fuzz.trimf(self.payment_delay.universe, [40, 50, 60, 60])

        # Надежность
        self.reliability['низкая'] = fuzz.trimf(self.reliability.universe, [0, 0, 40])
        self.reliability['средняя'] = fuzz.trimf(self.reliability.universe, [30, 50, 70])
        self.reliability['высокая'] = fuzz.trimf(self.reliability.universe, [60, 100, 100])

    def setup_rules(self):
        rules = [
            ctrl.Rule(self.delivery_delay['низкая'] & self.quality_defects['низкая'] & self.completeness['высокая'],
                      self.reliability['высокая']),
            ctrl.Rule(self.delivery_delay['высокая'] | self.quality_defects['высокая'] | self.completeness['низкая'],
                      self.reliability['низкая']),
            ctrl.Rule(self.price_deviation['выгодная'] & self.delivery_delay['низкая'] & self.quality_defects['низкая'],
                      self.reliability['высокая']),
            ctrl.Rule(self.price_deviation['невыгодная'] & self.quality_defects['средняя'], self.reliability['средняя']),
            ctrl.Rule(self.payment_delay['высокая'] & self.delivery_delay['средняя'], self.reliability['низкая']),
            ctrl.Rule(self.delivery_delay['средняя'] & self.quality_defects['средняя'] & self.completeness['средняя'],
                      self.reliability['средняя']),
            ctrl.Rule(self.completeness['высокая'] & self.quality_defects['низкая'] & self.payment_delay['низкая'],
                      self.reliability['высокая']),
            ctrl.Rule(self.delivery_delay['высокая'] & self.quality_defects['низкая'], self.reliability['средняя']),
            ctrl.Rule(self.price_deviation['выгодная'] & self.completeness['высокая'] & self.delivery_delay['низкая'],
                      self.reliability['высокая']),
            ctrl.Rule(self.completeness['низкая'] & self.quality_defects['высокая'], self.reliability['низкая']),
            ctrl.Rule(self.payment_delay['средняя'] & self.price_deviation['нормальная'], self.reliability['средняя']),
            ctrl.Rule(self.delivery_delay['низкая'] & self.completeness['высокая'] & self.price_deviation['нормальная'],
                      self.reliability['высокая'])
        ]

        self.reliability_ctrl = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.reliability_ctrl)

    def evaluation(self, delivery_delay: int, quality_defects: int, completeness: int, price_deviation: int, payment_delay: int) -> dict:
        self.simulation.input['delivery_delay'] = delivery_delay
        self.simulation.input['quality_defects'] = quality_defects
        self.simulation.input['completeness'] = completeness
        self.simulation.input['price_deviation'] = price_deviation
        self.simulation.input['payment_delay'] = payment_delay

        self.simulation.compute()

        score = self.simulation.output['reliability']

        deg_low = fuzz.interp_membership(self.reliability.universe, self.reliability['низкая'].mf, score)
        deg_med = fuzz.interp_membership(self.reliability.universe, self.reliability['средняя'].mf, score)
        deg_high = fuzz.interp_membership(self.reliability.universe, self.reliability['высокая'].mf, score)

        return {
            'reliability_index': round(score, 2),
            'deg_low': round(deg_low, 3),
            'deg_medium': round(deg_med, 3),
            'deg_high': round(deg_high, 3)
        }





