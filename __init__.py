from .src.rxn_fit import (
    Quantity, NonhomogenousQuantity, Reaction, Constraint, RateFit, FitResult,
    optimize, fit, visualize, from_json_input
)

__all__ = [
    'Quantity', 'NonhomogenousQuantity', 'Reaction', 'Constraint', 'RateFit', 
    'FitResult', 'optimize', 'fit', 'visualize', 'from_json_input'
]

# import pandas as pd
# import numpy as np
# from src import rxn_fit
# from src.quantity import Quantity
# from src.rxn_decode import Reaction
# t = np.array([0, 1, 2, 3, 4, 5, 6, 18, 24])
# df = pd.read_csv('data.csv', index_col=0).T
# compounds = ['DZN', 'DHD', 'THD', 'EQL']
# rxn = Reaction(compounds)
# rxn.add_steps([
#     'DZN = DHD',
#     'DHD = THD',
#     'THD = EQL'
# ])
# fr = rxn_fit.fit(
#     rxn,
#     Quantity(t, 'hr', 'time').convert('min'),
#     Quantity(df.to_numpy(), 'mM', 'molarity'),
#     integration_method='euler_i',
#     epoch_limit=100,
# )
# rxn_fit.visualize(
#     fr, 
#     Quantity(t, 'hr', 'time').convert('min'),
#     Quantity(df.to_numpy(), 'mM', 'molarity'),
#     rxn
# )
