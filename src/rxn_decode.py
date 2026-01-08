import copy
import pandas as pd
from .quantity import NonhomogenousQuantity, QUANTITY_TYPE


#NOTE: "A = " 꼴의 자기 생성/파괴 반응에 대해서 안정성 테스트 필요
class Reaction:
    '''
    Reaction.stoichiometry (num_reactions, num_compounds)
    '''
    def __init__(self, compounds:list):
        self.compounds = compounds
        self.stoichiometry = []
        self.steps = []
        
    def add_step(self, step:str):
        '''
        e.g.) step = "2A + B = C + 3D"
        '''
        compounds, stoichiometries = decode_string_step(step)
        expr_stoichiometry = [0] * len(self.compounds)        
        for compound, stoichiometry in zip(compounds, stoichiometries):
            if compound:
                expr_stoichiometry[self.compounds.index(compound)] += stoichiometry
            else:
                continue
        self.stoichiometry.append(expr_stoichiometry)
        self.steps.append(step)
        
    def add_steps(self, steps:list):
        for step in steps:
            self.add_step(step)
            
    def __call__(self):
        return pd.DataFrame(
            self.stoichiometry, 
            index=self.steps,
            columns=[f'\u03bd_i({comp})' for comp in self.compounds],
        )


def decode_string_step(step:str):
    '''
    return two list of compounds and stoichiometries in step
    '''
    step_split = step.strip().split('=')
    if len(step_split) != 2:
        raise RuntimeError(step_split)
    reactants, products = step_split
    reactants = [reactant.strip() for reactant in reactants.strip().split('+')]
    products = [product.strip() for product in products.strip().split('+')]
    stoichiometries = []
    compounds = []
    for sign, side in zip([-1, 1], [reactants, products]):
        for term in side:
            found_nonnumeric = False
            stoichio = []
            compound = []
            for letter in term:
                if not found_nonnumeric and letter.isnumeric():
                    stoichio.append(letter)
                else:
                    found_nonnumeric = True
                    compound.append(letter)
            if stoichio:
                stoichio = sign * int(''.join(stoichio))
            else:
                stoichio = sign
            compound = ''.join(compound).strip()
            stoichiometries.append(stoichio)
            compounds.append(compound)
    return compounds, stoichiometries
  
def rxn_from_steps(steps:list):
    compounds = set()
    for step in steps:
        compounds |= set(decode_string_step(step)[0])
    rxn = Reaction(list(compounds))
    rxn.add_steps(steps)
    return rxn


class Constraint:
    def __init__(self, compounds:list, conc_unit:str='mM'):
        self.compounds = compounds
        self.values = pd.DataFrame(columns=compounds+['const', 'unit'])
        self.set_conc_unit(conc_unit)
    
    def set_conc_unit(self, conc_unit:str):
        if conc_unit in QUANTITY_TYPE['molarity']:
            self.unit = conc_unit
        else:
            raise ValueError
        
    def add_condition(self, condition:str, unit:str=None):
        '''
        e.g.) condition = "A + B = 100"
        '''
        if unit is None:
            unit = self.unit
        compounds, coefficients = decode_condition(condition, unit)
        self.values.loc[self.values.shape[0], compounds] = coefficients
        
    def add_conditions(self, conditions:list, units:list):
        if isinstance(units, str):
            units = len(conditions) * [units]
        elif len(units) == 0:
            units = len(conditions) * units
        for condition, unit in zip(conditions, units):
            self.add_condition(condition, unit)
        
    def __call__(self, unit:str=None):
        if unit is None:
            unit = self.unit
        values = self.values.copy()
        const = NonhomogenousQuantity(
            values['const'].to_numpy(), values['unit'].to_numpy()
        )
        values['const'] = const.convert({'molarity': unit}).to_quantity()
        values['unit'] = unit
        values = values.infer_objects(copy=False).fillna(0.)
        return values
        

def _tokenize_by(x:str, str_list:list):
    for s in str_list:
        x = f' {s} '.join([
            token.strip() for token in x.strip().split(s)
        ])
    tokens = x.split()
    return tokens

def _collate_minus(tokens:list, remove_plus:bool=True):
    tokens = copy.deepcopy(tokens)
    for i, s in enumerate(tokens):
        if s == '-':
            tokens[i] = '+'
            tokens[i + 1] = f'-{tokens[i + 1]}'
    if remove_plus:
        while '+' in tokens:
            tokens.remove('+')
    return tokens

def _isnumerical(x:str):
    if isinstance(x, str):
        return x.isnumeric() or x in ['.', '+', '-']
    else:
        raise ValueError
    
def _parse_term(term:str):
    for i, char in enumerate(term):
        isnum = _isnumerical(char)
        if not isnum:
            break
    if isnum:
        i += 1
    coefficient, variable = term[:i], term[i:]
    return coefficient, variable
    
def _parse_expression(expr:str):
    tokens = _tokenize_by(expr, ['+', '-'])
    tokens = _collate_minus(tokens)
    vars = []
    coeffs = []
    for token in tokens:
        coeff, var = _parse_term(token)
        if coeff:
            coeff = float(coeff)
        else:
            coeff = 1.
        if var:
            vars.append(var)
            coeffs.append(coeff)
        elif 'const' in vars:
            coeffs[var.index('coeffs')] += coeff
        else:
            vars.append('const')
            coeffs.append(coeff)
    return vars, coeffs

def decode_condition(condition:str, unit:str):
    condition_split = condition.strip().split('=')
    if len(condition_split) != 2:
        raise RuntimeError(condition_split)
    left_expr, right_expr = condition_split
    variables, coeffs = _parse_expression(left_expr)
    r_vars, r_coeffs = _parse_expression(right_expr)
    for var, coeff in zip(r_vars, r_coeffs):
        if var in variables:
            coeffs[variables.index(var)] -= coeff
        else:
            variables.append(var)
            coeffs.append(-coeff)
    variables.append('unit')
    coeffs.append(unit)
    return variables, coeffs