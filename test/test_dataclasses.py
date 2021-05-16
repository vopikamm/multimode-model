""" Test the behaviour of the dataclasses.
"""
import sys
import os
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


from shallow_water_eqs import Parameters


class TestParameters:

    def test_default_values(self):
        p = Parameters()
        defaults = {
            'f': 0.,
            'g': 9.81,
            'beta': 2./(24*3600),
            'H': 1000.,
            'dt': 8.,
            't_0': 0.,
            't_end': 3600.,
            'write': 20.,
        }
        for var, val in defaults.items():
            assert p.__getattribute__(var) == val