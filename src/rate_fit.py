import torch
import numpy as np
from .rxn_decode import Constraint

def inv_softplus(x:torch.Tensor, beta=1, threshold=20):
    if (x <= 0).any():
        raise RuntimeError(x)
    mask = x <= threshold / beta
    out = torch.empty_like(x)
    out[mask] = torch.log(torch.expm1(beta * x[mask])) / beta
    out[~mask] = x[~mask]
    return out

def pow(x:torch.Tensor, n:torch.Tensor):
    return torch.exp(n * torch.log(x))

def smooth_clamp(x, min, max, beta=0.1):
    return min + (max - min) * torch.sigmoid(
        beta * (x - (min + max) / 2)
    )

def inv_smooth_clamp(y, min, max, beta=0.1):
    return (min + max) / 2 + torch.logit(
        (y - min) / (max - min)
    ) / beta

class RateFit(torch.nn.Module):
    def __init__(
        self, 
        stoichiometry:torch.Tensor, 
        init_compound_concentrations:torch.Tensor,
        t:torch.Tensor,
        k_pi:torch.Tensor=None,
        k_mi:torch.Tensor=None,
        resolution:int=100,
        *args, **kwargs
    ):
        '''
        stoichiometry :                 (num_reactions, num_compounds)
        init_compound_concentrations :  (num_compounds,)
        t :                             (num_time_points,)
        self.t :                        (<= num_time_points + resolution,)
        k_p1 | self.f :                 (num_reactions,)
        k_m1 | self.r :                 (num_reactions,)
        self.order :                    (num_reactions, 2 [forward, reverse])
        '''
        super().__init__(*args, **kwargs)
        stoichiometry = torch.as_tensor(
            stoichiometry, dtype=torch.float32
        )
        num_reactions, num_compounds = stoichiometry.shape
        init_compound_concentrations = torch.as_tensor(
            init_compound_concentrations, dtype=torch.float32
        )
        t = torch.as_tensor(t, dtype=torch.float32)
        self.t = torch.linspace(t.min(), t.max(), resolution, device=t.device)
        self.init_concentrations = init_compound_concentrations.reshape(-1)
        if stoichiometry.ndim != 2:
            raise IndexError(stoichiometry.ndim)
        self.stoichiometry = stoichiometry
        stoi_f = stoichiometry.detach().clone()
        stoi_f[stoichiometry < 0] = 0
        stoi_r = stoichiometry.detach().clone()
        stoi_r[stoichiometry > 0] = 0
        order = torch.stack([stoi_f.sum(1), -stoi_r.sum(1)], dim=1)
        self.order = order
        # According to the characteristic time of reactions:
        # tau = 1 / k / C ** (order - 1)
        rxn_const_ub = t.shape[0] * 10 / t.diff().max() / init_compound_concentrations.sum(
        ) ** (
            order - 1
        )
        self.rxn_const_ub = rxn_const_ub
        if self.init_concentrations.shape[0] != num_compounds:
            raise ValueError(self.init_concentrations)
        f = -torch.ones(num_reactions) if k_pi is None else inv_smooth_clamp(
            torch.as_tensor(k_pi, dtype=torch.float32), 0, rxn_const_ub[:, 0]
        )
        r = -torch.ones(num_reactions) if k_mi is None else inv_smooth_clamp(
            torch.as_tensor(k_mi, dtype=torch.float32), 0, rxn_const_ub[:, 1]
        )
        if f.shape[0] != num_reactions or r.shape[0] != num_reactions:
            raise ValueError(f.shape, r.shape)
        self.f = torch.nn.Parameter(f)
        self.r = torch.nn.Parameter(r)
    
    def reaction_coeff(self):
        '''
        rxn coeff return shape of (num_reactions, 2 [forward, reverse])
        '''
        return torch.stack([
            smooth_clamp(
                self.f.detach().clone(), 0, self.rxn_const_ub[:, 0].detach()
            ),
            smooth_clamp(
                self.r.detach().clone(), 0, self.rxn_const_ub[:, 1].detach()
            ),
        ], dim=1)

    def _set_scale(self, t_scale, conc_scale):
        self.scales = {
            'time': t_scale,
            'concentration': conc_scale,
        }

    @staticmethod
    def _get_dCdt(C_j, nu_ij, k_pi, k_mi):
        '''
        dC_j/dt = sum_i(
            nu_ij * (
                k_+i * prod_{j where nu_ij < 0} (
                    C_j **  - nu_ij    
                ) 
                - k_-i * prod_{j where nu_ij > 0} (
                    C_j ** nu_ij
                )
            )
        ) 
        '''
        C_j = C_j.reshape(1, -1)
        C_j_safe = torch.clamp(C_j, min=1e-8, max=1e3)
        forward_rxn = k_pi * torch.where(
            nu_ij < 0,
            pow(C_j_safe, -nu_ij), 
            torch.ones_like(nu_ij),
        ).prod(1, keepdim=True)
        reverse_rxn = k_mi * torch.where(
            nu_ij > 0, 
            pow(C_j_safe, nu_ij), 
            torch.ones_like(nu_ij), 
        ).prod(1, keepdim=True)
        return (nu_ij * (forward_rxn - reverse_rxn)).sum(0)
    
    def get_curves(self, method:str, is_implicit:bool=False):
        '''
        return curve of (num_time_points, num_compounds)
        '''
        dts = torch.diff(self.t)
        nu_ij = self.stoichiometry
        k_pi = smooth_clamp(self.f, 0, self.rxn_const_ub[:, 0]).reshape(-1, 1)
        k_mi = smooth_clamp(self.r, 0, self.rxn_const_ub[:, 1]).reshape(-1, 1)
        C_j = self.init_concentrations
        concentrations = [C_j.clone()]
        for dt in dts:
            if is_implicit:
                if method == 'euler':
                    C_j0 = C_j
                    # C_j1 = C_j0 + dt * self._get_dCdt(C_j0, nu_ij, k_pi, k_mi)
                    k1 = self._get_dCdt(C_j0,               nu_ij, k_pi, k_mi)
                    k2 = self._get_dCdt(C_j0 + dt * k1 / 2, nu_ij, k_pi, k_mi)
                    C_j1 = C_j0 + k2 * dt
                    def F(x):
                        return x - C_j0 - dt * self._get_dCdt(
                            x, nu_ij, k_pi, k_mi
                        )
                    for _ in range(5):
                        res = F(C_j1)
                        if torch.norm(res) < 1e-5 * torch.norm(C_j1):
                            break
                        J = torch.func.jacrev(F)(C_j1.detach().requires_grad_(True))
                        #Levenberg–Marquardt damping
                        J = J + 1e-7 * torch.eye(J.shape[0], device=J.device)
                        delta = torch.linalg.solve(J, -res)
                        C_j1 = C_j1 + delta
                    C_j = C_j1
                else:
                    raise RuntimeError(method)
            else:
                if method == 'euler':
                    dC_jdt = self._get_dCdt(C_j, nu_ij, k_pi, k_mi)
                    dC_j = dC_jdt * dt
                elif method == 'rk2':
                    k1 = self._get_dCdt(C_j,               nu_ij, k_pi, k_mi)
                    k2 = self._get_dCdt(C_j + dt * k1 / 2, nu_ij, k_pi, k_mi)
                    dC_j = k2 * dt
                elif method == 'rk4':
                    k1 = self._get_dCdt(C_j,               nu_ij, k_pi, k_mi)
                    k2 = self._get_dCdt(C_j + dt * k1 / 2, nu_ij, k_pi, k_mi)
                    k3 = self._get_dCdt(C_j + dt * k2 / 2, nu_ij, k_pi, k_mi)
                    k4 = self._get_dCdt(C_j + dt * k3,     nu_ij, k_pi, k_mi)
                    dC_j = (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt
                else:
                    raise RuntimeError(method)
                C_j = C_j + dC_j
            concentrations.append(C_j.reshape(-1))
        return torch.stack(concentrations)
    
    def predict(
        self, 
        t:torch.Tensor, 
        integration_method:str='euler', 
        is_learning:bool=False
    ):
        integration_method = integration_method.strip().split('_')
        if len(integration_method) == 1:
            method = integration_method[0].strip()
            is_implicit = False
        elif len(integration_method) == 2:
            method = integration_method[0].strip()
            if integration_method[1].strip() == 'i':
                is_implicit = True
            else:
                is_implicit = False
        else:
            raise ValueError
        t = t.reshape(-1, 1)
        t_dist_in_grid = (self.t.reshape(1, -1) - t).abs()
        i_argsort = t_dist_in_grid.argsort(1).T
        i1, i2 = i_argsort[:2]
        if is_learning:
            curve = self.get_curves(method, is_implicit)
        else:
            curve = self.get_curves(method, is_implicit).detach().clone()
        return self.interpolate(t, i1, i2, curve)
    
    def interpolate(
        self,
        t:torch.Tensor,
        i1:torch.Tensor,
        i2:torch.Tensor,
        curves:torch.Tensor,
    ):
        t1 = self.t[i1].reshape(-1, 1)
        t2 = self.t[i2].reshape(-1, 1)
        c1 = curves[i1]
        c2 = curves[i2]
        slope = (c2 - c1) / (t2 - t1)
        return slope * (t - t1) + c1
    
    def forward(self, t:torch.Tensor, integration_method:str='euler'):
        return self.predict(
            t, 
            integration_method=integration_method, 
            is_learning=True
        )
        
def squared_error(x:torch.Tensor, x_target:torch.Tensor):
    x = torch.as_tensor(x)
    x_target = torch.as_tensor(x_target)
    if x.shape != x_target.shape: 
        raise RuntimeError
    return torch.pow(x - x_target, 2)


def optimize(
    t:torch.Tensor, 
    concentrations:torch.Tensor,
    rate_fit:RateFit,
    constraint:Constraint=None,
    integration_method:str='euler_i',
    convergence_threshold:float=1e-5,
    loss_threshold:float=1e-7,
    print_loss=False,
    epoch_limit:int=None,
    lr:float=0.1,
):
    '''
    concentrations : (num_time_points, num_compounds)
    '''
    if not (isinstance(epoch_limit, (int)) or epoch_limit is None):
        raise RuntimeError(epoch_limit)
    t = torch.as_tensor(t, dtype=torch.float32).detach().clone()
    concentrations = torch.as_tensor(
        concentrations, dtype=torch.float32
    ).detach().clone()
    isnan_mask = concentrations.isnan()
    concentrations[isnan_mask] = 0
    optimizer = torch.optim.Adam(rate_fit.parameters(), lr=lr)
    losses = []
    epoch = -1
    while (len(losses) < 10
           or np.std(losses[-10:]) 
                / np.abs(np.mean(losses[-10:])) > convergence_threshold
           ):
        epoch += 1
        optimizer.zero_grad()
        pred_y = rate_fit(t, integration_method=integration_method)
        se = squared_error(pred_y, concentrations)
        se = se.masked_fill(isnan_mask, 0.0)
        rate_mse = se.sum() / (~isnan_mask).sum()
        rate_rmse = rate_mse ** 0.5
        if constraint is None:
            constraint_rmse = 0
        else:
            constraint_scale = (
                rate_fit.scales['concentrations'] 
                if 'scales' in rate_fit.__dict__ else
                1 
            )
            constraint_tensor = torch.tensor(
                constraint().iloc[:, :-1].to_numpy().T,
                device=concentrations.device
            )
            constraint_tensor[:, -1] /= constraint_scale
            constraint_error = torch.cat(
                [
                    concentrations, 
                    torch.ones(
                        (concentrations.shape[0], 1), device=concentrations.device
                    )    
                ], dim=1
            ) @ constraint_tensor
            constraint_rmse = (constraint_error ** 2).mean() ** 0.5
        loss = rate_rmse + constraint_rmse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rate_fit.parameters(), max_norm=10.)
        optimizer.step()
        losses.append(loss.item())
        if print_loss and epoch % 10 == 0: 
            print(f'epoch {epoch} | MSE: {loss.item()}')
        if loss < loss_threshold:
            break
        if (epoch_limit is not None) and epoch == epoch_limit:
            break

'''
오일러 방법의 안정화 조건:

다음 자코비안 (roundf_j/roundC_k)에 대해

J_ijk = round/roundC_k dCj/dt = sum_i(
    nu_ij * (
        k_+i * (
                -nu_ik * C_k ** (-nu_ik - 1)
        )_{where nu_ik < 0} * prod_{j where nu_ij < 0 and j != k} (
            C_j **  - nu_ij    
        ) 
        - 
        k_-i * (
            nu_ik * C_k ** (nu_ik - 1)
        )_{where nu_ik > 0} * prod_{j where nu_ij > 0 and j != k} (
            C_j ** nu_ij
        )
    )
)
EVD: J = V Lambda V^-1
Lambda 의 모든 성분 lambda_i에 대해

max_i (|1 + dt * lambda_i|) < 1

이어야, dt로 한 오일러 방법은 안정한 해에 수렴한다.

또한 감쇠 조건이 필요하기에 lambda_i = a + jb의 실수부 a < 0 이어야 한다.
(아마도 대부분의 화학반응식은 그럴것)
'''