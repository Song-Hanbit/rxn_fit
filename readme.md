# rxn_fit :test_tube:
A rate constant optimizer utilizing numerical solutions for simultaneous ODEs of chemical reactions.

# Quick start
### 1. Run with a json input file
The quickest way to start is by using `from_json_input()`.
```python
from rxn_fit import from_json_input
from_json_input('rxn_fit/input.json')
```
The json input file contains a mapping of arguments and data. Available arguments are:

- `data`: *string*, the csv file containing reaction profiles with a shape of (*num_compounds*, *num_time_points*) 
- `reactions`: *list*, a list of reaction steps
- `epoch`: *integer* (defaults to `1000`), the maximum number of iterations for ODE fitting
- `resolution`: *integer* (defaults to `100`), the number of time points for the numerical solution optimization
- `method`: *string*, (defaults to `'euler_i'`), integration method for the numerical solution. Available methods are *euler_i: implicit Euler, euler: explicit Euler, rk2: 2nd explicit Runge-Kutta, rk4: 4th explicit Runge-Kutta*.
- `units`: *dictionary*, (defaults to `{"time": "min", "molarity": "mM"}`), the unit system for the reaction.

An examplary json at `rxn_fit/input.json` is:
```json
{
    "data": "rxn_fit/data.csv",
    "reactions": [
        "A = B",
        "B = C",
        "C = D",
    ],
    "epoch": 100
}
```
The shape of data csv file should be (*num_compounds*, *num_time_points*) starting at **B2**. The headers in row **A** are time points, and the indices in the column **1** are the compounds.

`rxn_fit/data.csv`:
||0|60|120|180|240|300|360|1080|1440|
|---|---|---|---|---|---|---|---|---|---|
|A|4.78906206467155|4.01836320051025|3.16447478676158|2.72632158304059|2.13086631197345|1.8589879382761|1.30269926624492|0|0.000920504206298667|
|B|0.00010579442995|0.233202562244365|0.448783824001822|0.525901536222731|0.435268545639069|0.325631466661621|0.165354028905519|0.0342681078700033|0.0524396199172947|
|C||||||||||
|D|0|0.215041772652451|0.635245113973011|1.21099620189165|1.80093328507804|2.41133575075487|2.75845549045259|4.27372957874223|4.26434754332108|


---
### 2. Run with a pure script
You can run a script by importing `rxn_fit` in an interactive shell.

Example:
```python
import pandas as pd
from rxn_fit import Quantity, Reaction, fit, visualize
df = pd.read_csv('rxn_fit/data.csv', index_col=0).T
compounds = df.columns.tolist()
t = df.index.astype(float).tolist()
rxn = Reaction(compounds)
rxn.add_steps([
    'A = B',
    'B = C',
    'C = D'
])
result = fit(
    rxn,
    Quantity(t, 'min', 'time'),
    Quantity(df.to_numpy(), 'mM', 'molarity'),
    integration_method='euler_i',
    epoch_limit=100,
)
visualize(
    result, 
    Quantity(t, 'min', 'time'),
    Quantity(df.to_numpy(), 'mM', 'molarity'),
    rxn
)
```
---
### 3. Run on CLI
A shell command executes the `from_json_input()` method.
```sh
$ python3 rxn_fit -i rxn_fit/input.json
```

# Detailed usage
Importing this package provides interactive tools for the analysis of chemical equations.

```python
import rxn_fit
```
---
### 1. Handling physical values
`Quantity` class is a subclass of `numpy.ndarray` with an associated physical unit. It accepts an array and a string of unit.
```python
Quantity(
    array_like: any,
    unit: str,
    quantity_type: str = None,
)
```
Providing `quantity_type` reduces validation and search time for the `unit` based on the `rxn_fit.src.quantity.QUANTITY_TYPE`. Currently supported quantity types are `'time', 'mass', 'length', 'mol', 'volume', 'molarity'`.
#### Example:
```python
concentrations = rxn_fit.Quantity(
    [[1, 2], [3, 4]],
    'mM',
    'molarity'
)
```
Unit conversion can be achieved by `Quantity.convert()`.
```python
(method) def convert(unit: str) -> Quantity
```
#### Example:
```python
conc_lambda = concentrations.convert('uM')
```
If an array of values does not share same dimension or unit, `NonhomogenousQuantity` is useful. It accepts an array of values and an array of units.
```python
NonhomogenousQuantity(
    array_like: any,
    units: any,
)
```
`units` will be broadcast according to the shape of `array_like` using the same rules of broadcasting in NumPy.
#### Example:
```python
conc_and_time = rxn_fit.NonhomogenousQuantity(
    [[1, 2], [3, 4]],
    [['mM', 'uM'], ['sec', 'min']],
)
```
Unit conversion requires a mapping of quantity_types and units.
```python
(method) def convert(unit_map: dict = {}) -> NonhomogenousQuantity
```
#### Example:
```python
converted = conc_and_time.convert({'molarity': 'mM', 'time': 'min'})
```
If the contained units turn out to be a single unique unit, a `NonhomogenousQuantity` instance can be converted to a `Quantity` instance using `NonhomogenousQuantity.to_quantity()`.

#### Example:
```python
concentrations = rxn_fit.NonhomogenousQuantity(
    [[1, 2], [3, 4]],
    [['mM', 'uM'], ['M', 'nM']],    
)
unique_conc = concentrations.convert({'molarity': 'mM'})
unique_conc = unique_conc.to_quantity()
```
---
### 2. Definition of chemical equations
Chemical equations can be encoded using the `Reaction` class. The main purpose of this class is to extract the stoichiometric array from a set of simultaneous chemical equations.
```python
Reaction(compounds: list)
```
Initially, the compounds participating in this system are defined. The `compounds` argument should consist of strings representing chemical species. 
> Currently, the contained strings in `compounds` should only contain alphabetic characters.

Adding elementary reaction steps can be done using either `Reaction.add_step()` or `Reaction.add_steps()` method.
```python
(method) def add_step(step: str)
(method) def add_steps(steps: list)
```
A `step` should consist only of strings from `Reaction.compounds`, integer stoichiometric coefficients, `'=', '+'`, or space. `'='` denotes a bidirectional reaction.
#### Example:
```python
rxn = rxn_fit.Reaction(['A', 'B', 'C', 'D', 'E'])
rxn.add_step('A + 2B = C')
rxn.add_steps([
    'C = 2D',
    '2A + 2D = E',
])
```
Calling the `Reaction` instance returns a stoichiometric array of `pandas.DataFrame` shaped (*num_steps*, *num_compounds*).
#### Example:
```python
rxn()
```
---
### 3. Adding constraints *(in test, optional)*
Constraints to satisfy physical rules, like mass conservation, can be added using the `Constraint` class. One example is the conservation of racemic compounds under stereospecific reaction conditions when detection methods for enantiomers are limited. The constraint can be expressed as linear simultaneous equations:
> **A @ C - B = 0**, where **A** represents the coefficients of compound concentrations, **C**, and **B** is total concentration of **A @ C**.
```python
Constraint(compounds: list, conc_unit: str = 'mM')
```
`compounds` should be the same as `Reaction.compounds` and `conc_unit` defines the unit of **B**, setting `Constraint.unit`.

Adding constraining conditions can be done using either the `Constraint.add_condition()` or `Constraint.add_conditions()` method.
```python
(method) def add_condition(condition: str, unit: str = None)
(method) def add_conditions(conditions: list, units: list)
```
A `condition` should consist only of the strings in `Constraint.compounds`,  coefficients (**A**) or constants (**B**) of float values, `'=', '+', '-'` or space.
#### Example:
```python
constraints = rxn_fit.Constraint(['A', 'B', 'C', 'D', 'E'], 'mM')
constraints.add_condition('A + B = 10', 'uM')
constraints.add_conditions(
    ['C = D + 10', 'D - E - 20 = 0'], ['mM', 'nM']
)
```
Calling the `Constraint` instance returns an array of `pandas.DataFrame` shaped (*num_conditions*, *num_compounds + 2*)
representing the simultaneous equations. The columns consist of `self.compounds + ['const', 'unit']` where the `'const'` column is **-B** and the `'unit'` column is its unit. 
#### Example:
```python
constraints()
```
---
### 4. Optimization for the numerical solution of chemical equations' ODEs
The `RateFit` class and the `optimize` function are key engines for optimizing rate constants using the `pytorch` package. One iteration (epoch) of optimization executes the following procedure:
1. **Numerical integration** using the current solution (i.e. estimated rate constants) along the time axis with a given resolution
2. **Linear interpolation** of the integration result to obtain estimated data points for comparison with actual data points
3. Mean squared error between the data acts as a **loss** (MSELoss) to back-propagate the gradients.
4. The solution is **updated** by the gradients.

However, one can simply use `fit` function which wraps `RateFit` and `optimize`.
```python
(function) def fit(
    reaction: Reaction,
    t: Quantity,
    conc: Quantity,
    initial_conc: Quantity = None,
    constraint: Constraint = None,
    resolution: int = 100,
    integration_method: str = 'euler_i',
    epoch_limit: int = 1000,
    verbose: bool = True
) -> FitResult
```
- `reaction`: *Reaction* instance containing elementary reaction steps
- `t`: *Quantity*, time points of experimental data, shape of (*num_time_points*,)
- `conc`: *Quantity*, concentration of compounds at the time points in the experimental data, with a shape of (*num_time_points*, *num_compounds*). Missing values can be annotated by `np.nan`. The ground truth for MSELoss.
- `initial_conc`: *Quantity* (defaults to `None`), initial concentration of compounds at `t[0]`, with a shape of (*num_compounds*,). If it is None, `conc[0]` will be used.
- `constraint`: (optional, defaults to `None`), *Constraint* instance containing mass balance conditions as linear simultaneous equations. If it is not `None`, the MSELoss of these equations will be added to the overall loss.
- `resolution`: *integer* (defaults to `100`), the number of time points used for numerical analysis of ODEs
- `integration_method`: *string* (defaults to `'euler_i'`), integration method for the numerical solution. Available methods are: 
    - euler_i: implicit Euler 
    - euler: explicit Euler 
    - rk2: 2nd explicit Runge-Kutta
    - rk4: 4th explicit Runge-Kutta
- `epoch_limit`: *integer* (defaults to `1000`), the maximum epoch for optimization
- `verbose`: *boolean* (defaults to `True`), whether to be verbose

The `FitResult` class contains the result of optimization done by `fit()`. Its primary role is to provide optimized rate constants as `FitResult.coeff`, but it also provide the time points used in the integration (matching the `resolution` argument used in `fit()`) as `FitResult.t` and the predicted concentrations at those times as `FitResult.curve`. It also contains the `RateFit` instance used in `fit()` as `FitResult.rate_fit`.

---
### 5. Visualization of the solution
The return value of `fit()`, `FitResult`, provides rate constants with their repective units. You can also inspect the predicted concentration profiles based on the optimized rate constants using `visualize` function.
```python
(function) def visualize(
    fit_result: FitResult,
    t_obs: Quantity,
    curve_obs: Quantity,
    rxn: Reaction,
    units: dict = { 'time': 'min','molarity': 'mM' }
) -> pandas.DataFrame
```
- `fit_result`: *FitResult* instance returned by `fit()`
- `t_obs`: *Quantity*, time points of experimental data, with a shape of (*num_time_points*,)
- `curve_obs`: *Quantity*, concentration profile of the compounds at the time points of experimental data, with a shape of (*num_time_points*, *num_compounds*)
- `rxn`: *Reaction* instance used at `fit()`
- `units`: *dictionary* (defaults to `{'time': 'min', 'molarity': 'mM'}`), the unit system of the reaction

It will display a `bokeh` plot within the notebook if run on the Jupyter Notebook; otherwise, a web browser page will open for the plot. The returned `pd.DataFrame` contains the stoichiometric coefficients of the elementary reactions defined in `rxn`, as well as their foward and reverse rate constants in the columns of `'k_i'` and `'k_-i'`, respectively.