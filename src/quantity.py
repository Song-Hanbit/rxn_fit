import numpy as np

IDENTITY = {'': 1e0}

PREFIX = {
    'kilo':  1e+3,
    'deci':  1e-1,
    'centi': 1e-2,
    'milli': 1e-3,
    'micro': 1e-6,
    'nano':  1e-9,
    'pico':  1e-12,
}
PREFIX.update(IDENTITY)

PREFIX_1 = {
    'k': 1e+3,
    'd': 1e-1,
    'c': 1e-2,
    'm': 1e-3,
    'u': 1e-6,
    'n': 1e-9,
    'p': 1e-12,
}
PREFIX_1.update(IDENTITY)

def _unit_multiply(unit1, unit2, sep=''):
    combined = {}
    for k1, v1 in unit1.items():
        for k2, v2 in unit2.items():
            combined[f'{k1}{sep}{k2}'] = 10 ** (np.log10(v1) + np.log10(v2))
    return combined

def _unit_divide(unit1, unit2, sep='/'):
    combined = {}
    for k1, v1 in unit1.items():
        for k2, v2 in unit2.items():
            combined[f'{k1}{sep}{k2}'] = 10 ** (np.log10(v1) - np.log10(v2))
    return combined

def _unit_power(unit, power=1, sep='^'):
    powered = {}
    for k, v in unit.items():
        powered[f'{k}{sep}{power}'] = 10 ** (power * np.log10(v))
    return powered
    
def _parentheses(unit):
    par = {}
    for k, v in unit.items():
        par[f'({k})'] = v
    return par
    
TIME_UNIT = {
    'year':     60 * 60 * 24 * 7 * 365,
    'yr':       60 * 60 * 24 * 7 * 365,
    'y':        60 * 60 * 24 * 7 * 365,
    'week':     60 * 60 * 24 * 7,
    'day':      60 * 60 * 24,
    'd':        60 * 60 * 24,
    'hour':     60 * 60,
    'hr':       60 * 60,
    'h':        60 * 60,
    'minute':   60,
    'min':      60,
}
TIME_UNIT.update(_unit_multiply(PREFIX, {'second': 1e0}))
TIME_UNIT.update(_unit_multiply(PREFIX, {'sec': 1e0}))
TIME_UNIT.update(_unit_multiply(PREFIX_1, {'s': 1e0}))

MASS_UNIT = {
    'tonne':    1e3,
    'ton':      1e3,
    't':        1e3
}
MASS_UNIT.update(_unit_multiply(PREFIX, {'gram': 1e0}))
MASS_UNIT.update(_unit_multiply(PREFIX_1, {'g': 1e0}))

LENGTH_UNIT = _unit_multiply(PREFIX, {'meter': 1e0})
LENGTH_UNIT.update(_unit_multiply(PREFIX_1, {'m': 1e0}))

MOL_UNIT = _unit_multiply(PREFIX_1, {'mol': 1e0})
MOL_UNIT.update(_unit_multiply(PREFIX, {'mol': 1e0}))

VOLUME_UNIT = _unit_power(LENGTH_UNIT, 3)
VOLUME_UNIT.update(_unit_multiply(PREFIX_1, {'L': 1e-3}))
VOLUME_UNIT.update(_unit_multiply(PREFIX, {'liter': 1e-3}))
VOLUME_UNIT.update({'cc': 1e-6})

MOLARITY_UNIT = _unit_divide(MOL_UNIT, VOLUME_UNIT)
MOLARITY_UNIT.update(_unit_multiply(PREFIX_1, {'M': 1e3}))

QUANTITY_TYPE = {
    'time':     TIME_UNIT,
    'mass':     MASS_UNIT,
    'length':   LENGTH_UNIT,
    'mol':      MOL_UNIT,
    'volume':   VOLUME_UNIT,
    'molarity': MOLARITY_UNIT,
}

def add_nth_rate_const(n, is_elementary=True):
    if is_elementary:
        n = int(n)    
    q_type = f'{n}_rate'
    if 'reaction' not in QUANTITY_TYPE:
        zeroth = _unit_divide(MOLARITY_UNIT, TIME_UNIT)
        QUANTITY_TYPE['reaction'] = zeroth
        QUANTITY_TYPE['0_rate'] = zeroth
    if n < 0:
        raise ValueError(n)
    elif n == 0:
        return
    elif n == 1:
        units = _unit_divide(IDENTITY, TIME_UNIT)
    else:
        units = _unit_multiply(
            _unit_power(
                _parentheses(MOLARITY_UNIT), -(n - 1)
            ),
            _unit_power(TIME_UNIT, -1)
        )
    QUANTITY_TYPE[q_type] = units

def check_unit_type(unit:str, quantity_type:str=None):
    '''
    check or find the quantity type of unit
    '''
    unit = ''.join(unit.split())
    q_type = None
    if quantity_type is None:
        for q_type, units in QUANTITY_TYPE.items():
            if unit in units:
                return unit, q_type
        raise ValueError(f'Unknown unit: {unit}')
    else:
        if quantity_type not in QUANTITY_TYPE:
            raise RuntimeError(quantity_type)
        q_type = quantity_type
    return unit, q_type


class Quantity(np.ndarray):
    def __new__(cls, array_like:np.ndarray, unit:str, quantity_type:str=None):
        obj = np.asarray(array_like).view(cls)
        unit, q_type = check_unit_type(unit, quantity_type)
        obj.unit = unit
        obj.q_type = q_type
        return obj

    def __array_finalize__(self, obj):
        if obj is None: 
            return
        self.unit = getattr(obj, 'unit', None)
        self.q_type = getattr(obj, 'q_type', None)
        
    def convert(self, unit:str):
        if unit not in QUANTITY_TYPE[self.q_type]:
            raise ValueError(f'"{unit}" is not a unit of {self.q_type}')
        conversion = QUANTITY_TYPE[self.q_type]
        converted = self * conversion[self.unit] / conversion[unit]
        converted.unit = unit
        return converted
    
    def __repr__(self):
        return f'{super().__repr__()}\n{self.q_type} [{self.unit}]'
    
    def __str__(self):
        return f'{super().__str__()}\n{self.q_type} [{self.unit}]'


class NonhomogenousQuantity(np.ndarray):
    def __new__(cls, array_like:np.ndarray, units:np.ndarray):
        obj = np.asarray(array_like).view(cls)
        units = np.asarray(units, dtype=str)
        units = np.broadcast_to(units, obj.shape)
        obj.units = units
        return obj

    def __array_finalize__(self, obj):
        if obj is None: 
            return
        self.units = getattr(obj, 'units', None)
    
    def copy(self, order='C'):
        arr_copy = super().copy(order)
        return NonhomogenousQuantity(arr_copy, self.units.copy())

    def convert(self, unit_map:dict={}):
        '''
        unit_map : dict[quantity_type: unit]
        '''
        for q_type, unit in unit_map.items():
            if q_type not in QUANTITY_TYPE:
                raise ValueError(f'{q_type} is not in QUANTITY_TYPE')
            if unit not in QUANTITY_TYPE[q_type]:
                raise ValueError(f'{unit} is not a unit of {q_type}')
        unique_units = np.unique(self.units)
        converted = self.copy()
        if not unit_map:
            return converted
        elif unique_units.size == 0:
            raise RuntimeError
        elif len(unit_map) > unique_units.size:
            raise RuntimeError
        else:
            # 1 <= n(unit_map) <= n(unique_units)
            unit_array = converted.units.copy()
            for unit in unique_units:
                q_type = check_unit_type(unit)[1]
                if q_type in unit_map:
                    target_unit = unit_map[q_type]
                    conversion_table = QUANTITY_TYPE[q_type]
                    unit_mask = self.units == unit
                    converted[
                        unit_mask    
                    ] = self[
                        unit_mask    
                    ] * conversion_table[unit] / conversion_table[target_unit]
                    unit_array[unit_mask] = target_unit
            converted = NonhomogenousQuantity(converted, unit_array)
            return converted
    
    def to_quantity(self):
        units = np.unique(self.units)
        if units.size == 1:
            return Quantity(self, units.item())
        else:
            raise RuntimeError('Array with only one unit can be converted to Quantity')
    
    def _value_with_unit(self, len_print=11):
        abs_val = np.abs(self)
        float_mask = (abs_val == 0) | (
            (abs_val >= 10 ** -((len_print - 3) // 2)) & 
            (abs_val < 10 ** ((len_print - 3) // 2 + 1))
        )
        values = np.where(
            float_mask,
            np.char.mod(f'%{len_print}.{(len_print - 3) // 2}f', self),
            np.char.mod(f'%{len_print}.{(len_print - 3) // 2}e', self)
        )
        return np.char.rjust(
            np.char.add(values, np.char.add(' ', self.units)),
            12 + np.char.str_len(self.units).max(),
            ' '    
        )
    
    def __str__(self, len_print=11):
        return self._value_with_unit(len_print).__str__()
    
    def __repr__(self, len_print=11):
        return self.__str__(len_print)


if __name__ == '__main__':
    print('duplication test:')
    units = list(QUANTITY_TYPE.values())
    for i, us1 in enumerate(units):
        ui = list(range(len(units)))
        ui.remove(i)
        for i in ui:
            for k1 in us1:
                if k1 in units[i]: 
                    print(k1)
                

# import pandas as pd
# def csv_to_quantity(csv, unit_target, quantity_type=None):
#     unit_target, quantity_type = check_unit_type(unit_target, quantity_type)
#     if quantity_type is None:
#         raise RuntimeError
#     csv = pd.read_csv(csv)
#     qs = []
#     for value, unit in zip(csv.value.astype(float), csv.unit):
#         unit, q_type = check_unit_type(unit)
#         if q_type != quantity_type:
#             raise ValueError(value, q_type)
#         qs.append(Quantity(value, unit, q_type).convert(unit_target))
#     return Quantity(qs, unit_target, quantity_type)