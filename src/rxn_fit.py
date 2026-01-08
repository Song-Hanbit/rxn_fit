import json
import torch
import numpy as np
import pandas as pd
from IPython import get_ipython
from .quantity import Quantity, NonhomogenousQuantity, add_nth_rate_const
from .rxn_decode import Reaction, Constraint
from .rate_fit import optimize, RateFit
from . import quantity
from . import rxn_decode

DEVICE = 'cpu'

def in_ipython_notebook():
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
        return True
    else:
        return False

def in_ipython():
    if get_ipython() is None:
        return False
    else:
        return True

#TODO: 단위 변환 구현
class FitResult():
    def __init__(
        self, 
        coeff:NonhomogenousQuantity, 
        t:Quantity, 
        curve:Quantity, 
        rate_fit:RateFit,
    ):
        self.coeff = coeff
        self.t = t
        self.curve = curve
        self.rate_fit = rate_fit
        
    def __repr__(self):
        return self.coeff.__repr__()
        
    def __str__(self):
        return self.coeff.__str__()


def fit(
    reaction:Reaction,
    t:Quantity,
    conc:Quantity,
    initial_conc:Quantity=None,
    constraint:Constraint=None,
    resolution:int=100,
    integration_method='euler_i',
    epoch_limit=1000,
    verbose=True,
):
    '''
    t [=] min shape of (t,)
    conc [=] mM shape of (t, compounds)
    initial_conc [=] mM shape of (compounds,)
    
    Return:
    FitResult(
        rxn_coeffs: (num_reactions, 2 [forward, reverse]),
        t: (num_time_points),
        curve: (num_time_points, num_compounds),
        RateFit,
    )
    '''
    if isinstance(t, (Quantity, quantity.Quantity)):
        t = t.convert('min')
    else:
        t = Quantity(t, 'min', 'time')
    if isinstance(conc, (Quantity, quantity.Quantity)):
        conc = conc.convert('mM')
    else:
        conc = Quantity(conc, 'mM', 'molarity')
    if initial_conc is None:
        initial_conc = conc[0]
    elif isinstance(initial_conc, (Quantity, quantity.Quantity)):
        initial_conc = initial_conc.convert('mM')
    else:
        initial_conc = Quantity(initial_conc, 'mM', 'molarity')
    rxn_table = reaction()
    stoichiometry = torch.tensor(
        rxn_table.to_numpy(), dtype=torch.float32
    ).to(DEVICE)
    compounds = reaction.compounds
    reactions = rxn_table.index
    if constraint is not None:
        constraint.set_conc_unit('mM')
        if not (np.array(compounds) == np.array(constraint.compounds)).all():
            raise ValueError
    if verbose:
        print(
            'initial concentrations:',
            '\n'.join([
                f'{compound} : {init_conc} mM'
                for init_conc, compound in zip(
                    initial_conc, compounds
                )
            ]),
            'reactions:',
            *reactions,
            sep='\n'
        )
    t = torch.tensor(t, dtype=torch.float32).to(DEVICE)
    conc = torch.tensor(conc, dtype=torch.float32).to(DEVICE)
    initial_conc = torch.tensor(
        initial_conc, dtype=torch.float32    
    ).nan_to_num(0).to(DEVICE)

    # normalization
    t_scale = t[~t.isnan()].max().item()
    conc_scale = max(
        conc[~conc.isnan()].max().item(), 
        initial_conc[~initial_conc.isnan()].max().item()
    )
    t /= t_scale
    conc /= conc_scale
    initial_conc /= conc_scale

    rate_fit_ = RateFit(
        stoichiometry,
        initial_conc,
        t,
        resolution=resolution,
    ).to(DEVICE)
    rate_fit_._set_scale(t_scale, conc_scale)
    for order in rate_fit_.order.unique().tolist(): 
        add_nth_rate_const(order, is_elementary=True)
    optimize(
        t, 
        conc,
        rate_fit_,
        constraint=constraint,
        integration_method=integration_method,
        epoch_limit=epoch_limit,
        convergence_threshold=1e-10,
        lr=1.,
        print_loss=verbose,
    )
    dimensionless_rxn_coeffs = rate_fit_.reaction_coeff()
    # given reactions in rate_fit are elementary steps
    # denormalization
    rxn_orders = rate_fit_.order.to(dtype=torch.int).numpy()
    rxn_units = np.full_like(
        rxn_orders, '', dtype='<U6' 
        if rxn_orders.max() <= 1 else 
        f'<U{int(np.log10(rxn_orders.max() - 1)) + 12}'
    )
    for order in np.unique(rxn_orders):
        order_mask = rxn_orders == order
        if order == 0:
            dimensionless_rxn_coeffs[order_mask] *= conc_scale / t_scale
            rxn_units[order_mask] = 'mM/min'
        elif order == 1:
            dimensionless_rxn_coeffs[order_mask] /= t_scale
            rxn_units[order_mask] = '/min'
        else:
            dimensionless_rxn_coeffs[order_mask] *= conc_scale ** - (order - 1) * t_scale ** -1
            rxn_units[order_mask] = f'(mM)^-{order - 1}min^-1'
    rxn_coeffs = NonhomogenousQuantity(dimensionless_rxn_coeffs.tolist(), rxn_units)

    return FitResult(
        rxn_coeffs,
        Quantity(rate_fit_.t.numpy() * t_scale, 'min', 'time'),
        Quantity(
            rate_fit_.predict(
                rate_fit_.t, 
                integration_method='rk4',
                is_learning=False,    
            ).numpy() * conc_scale, 
            'mM', 
            'molarity',
        ),
        rate_fit_
    )

def visualize(
    fit_result:FitResult, 
    t_obs:Quantity, 
    curve_obs:Quantity,
    rxn:Reaction, 
    units:dict={
        'time':     'min',
        'molarity': 'mM',
    }
):
    '''
    t_obs       : (num_time_points,)
    curve_obs   : (num_time_points, num_compounds)
    '''
    if not isinstance(fit_result, FitResult):
        raise ValueError(type(fit_result))
    if not isinstance(units, dict):
        raise ValueError(type(units))
    time_unit = units['time']
    conc_unit = units['molarity']
    if t_obs is None:
        t_obs = Quantity(
            np.full_like(fit_result.t, np.nan), time_unit, 'time'
        )
    elif not isinstance(t_obs, (Quantity, quantity.Quantity)):
        t_obs = Quantity(t_obs, time_unit, 'time')
    t_obs.convert(time_unit)
    if curve_obs is None:
        curve_obs = Quantity(
            np.full_like(fit_result.curve, np.nan), conc_unit, 'molarity'
        )
    elif not isinstance(curve_obs, (Quantity, quantity.Quantity)):
        curve_obs = Quantity(curve_obs, conc_unit, 'molarity')
    curve_obs.convert(conc_unit)
    if (
        t_obs.ndim != 1 or 
        curve_obs.ndim != 2 or 
        t_obs.shape[0] != curve_obs.shape[0]
    ):
        raise ValueError(t_obs.shape, curve_obs.shape)
    if isinstance(rxn, (Reaction, rxn_decode.Reaction)):
        compounds = rxn.compounds
    else:
        raise ValueError(type(rxn))
    from bokeh.plotting import figure, show
    from bokeh.io import output_notebook
    from bokeh.palettes import viridis
    cmap = viridis(len(compounds))
    y_scale = np.nanmax(curve_obs).item()
    plot = figure(
        y_range=(0 - y_scale * 0.1, y_scale * 1.1)
    )
    plot.xaxis.axis_label = f't / {time_unit}'
    plot.yaxis.axis_label = f'concentration / {conc_unit}'
    plot.axis.axis_label_text_font_style = 'normal'
    curve_t = fit_result.t.convert(time_unit)
    curve = fit_result.curve.convert(conc_unit).T
    for conc, conc_obs, comp, color in zip(
        curve, 
        curve_obs.T, 
        compounds,
        cmap
    ):
        plot.line(curve_t, conc, color=color, legend_label=comp)
        plot.circle(
            t_obs, 
            conc_obs,
            color=color,
            radius=3.,
            radius_units='screen'
        )
    if in_ipython_notebook():
        output_notebook()
    show(plot)
    rxn_table = rxn()
    return pd.concat(
        [
            rxn_table, 
            pd.DataFrame(
                fit_result.coeff._value_with_unit(9), 
                index=rxn_table.index,
                columns=['k_i', 'k_-i']
            )    
        ], 
        axis=1
    )
    
def from_json_input(json_dir:str, verbose=True):
    args = {
        'resolution': 100,
        'method': 'euler_i',
        'epoch': 1000,
        'units': {
            'time': 'min',
            'molarity': 'mM'
        }
    }
    with open(json_dir, 'r') as json_dir:
        args.update(json.load(json_dir))
    data = pd.read_csv(args['data'], index_col=0)
    t = Quantity(
        data.columns.astype(float).to_list(), 
        args['units']['time'], 
        'time'
    ).convert('min')
    conc = Quantity(
        data.T, args['units']['molarity'], 'molarity'
    ).convert('mM')
    rxn = Reaction(data.index.to_list())
    rxn.add_steps(args['reactions'])
    fr = fit(
        rxn, 
        t, 
        conc, 
        None,
        None,
        args['resolution'], 
        args['method'], 
        args['epoch'], 
        verbose,
    )
    if in_ipython():
        return visualize(fr, t, conc, rxn, args['units'])
    else:
        print(
            visualize(fr, t, conc, rxn, args['units'])
        )
    