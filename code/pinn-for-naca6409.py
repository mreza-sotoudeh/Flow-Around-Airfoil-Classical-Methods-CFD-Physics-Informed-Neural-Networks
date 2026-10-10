# Full PINN (hybrid Adam + LBFGS) for steady incompressible Navier
Stokes 
# around NACA 6409. Solves for u, v, p. Produces 
pressure/velocity/streamline plots. 
# Requirements: numpy, matplotlib, torch 
 
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
from torch.autograd import grad 
from matplotlib.path import Path 
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import time 
import os 
import math 
 
# ---------------- reproducibility ---------------- 
seed = 1234 
np.random.seed(seed) 
torch.manual_seed(seed) 
if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(seed) 
 
# ---------------- device ---------------- 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("Device:", device) 
 
# ---------------- domain & flow params (adjustable) ---------------- 
chord = 1.0 
center_x = 1.0 
center_y = 0.0 
 
x_min, x_max = 0.25, 1.75 
y_min, y_max = -0.4, 0.4 
ub = np.array([x_max, y_max]) 
lb = np.array([x_min, y_min]) 
 
Re = 100.0 
nu = 1.0 / Re 
U_inf = 1.0 
 
# ---------------- sampling sizes (tune these for speed vs accuracy) ---------------- 
N_inlet = 600 
N_far = 400 
N_airfoil = 600 
 
# For debugging / quick test: lower these values (e.g., N_col=4000, 
N_refine=1000) 
N_col = 20000     # default collocation points (global) 
N_refine = 10000  # collocation points refined near airfoil 
 
# ---------------- NACA 6409 geometry ---------------- 
def naca_camber(x_coord, m=0.06, p=0.4, c=chord): 
    x = x_coord / c 
    if 0.0 <= x <= p: 
        return (m / p**2) * x * (2*p - x) 
    elif p < x <= 1.0: 
        return (m / (1 - p)**2) * ((1 - 2*p) + 2*p*x - x**2) 
    return 0.0 
 
def naca_thickness(x_coord, t=0.09, c=chord): 
    x = x_coord / c 
    return 5 * t * c * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 
0.2843*x**3 - 0.1015*x**4) 
 
def generate_naca_points(N_points=400, m=0.06, p=0.4, t=0.09): 
    n_half = N_points // 2 
    x_c = np.linspace(0, chord, n_half) 
    y_c = np.array([naca_camber(xi, m, p) for xi in x_c]) 
    y_t = np.array([naca_thickness(xi, t) for xi in x_c]) / 2.0 
    x_upper = x_c 
    y_upper = y_c + y_t 
    x_lower = x_c 
    y_lower = y_c - y_t 
    x_naca = np.concatenate([x_upper, x_lower[::-1]]) 
    y_naca = np.concatenate([y_upper, y_lower[::-1]]) 
    x_naca = x_naca + (center_x - 0.5*chord) 
    y_naca = y_naca + center_y 
    return np.stack([x_naca, y_naca], axis=1) 
 
def is_inside_airfoil(xy, airfoil_points): 
    path = Path(airfoil_points) 
    return path.contains_points(xy) 
 
# ---------------- sampling utils ---------------- 
def sample_boundary_points(alpha_deg): 
    alpha = np.deg2rad(alpha_deg) 
    Vx = U_inf * np.cos(alpha) 
    Vy = U_inf * np.sin(alpha) 
    # inlet 
    ys = np.linspace(y_min, y_max, N_inlet) 
    inlet_xy = np.vstack([x_min*np.ones_like(ys), ys]).T 
    inlet_uv = np.vstack([Vx*np.ones_like(ys), 
Vy*np.ones_like(ys)]).T 
    # far field: top, bottom, outlet 
    nf = max(3, N_far // 3) 
    xs_top = np.linspace(x_min, x_max, nf) 
    top_xy = np.vstack([xs_top, y_max*np.ones_like(xs_top)]).T 
    bot_xy = np.vstack([xs_top, y_min*np.ones_like(xs_top)]).T 
    ys_out = np.linspace(y_min, y_max, nf) 
    out_xy = np.vstack([x_max*np.ones_like(ys_out), ys_out]).T 
    far_xy = np.vstack([top_xy, bot_xy, out_xy]) 
    far_uv = np.tile(np.array([Vx, Vy]), (far_xy.shape[0],1)) 
    # airfoil surface 
    airfoil_pts = generate_naca_points(N_airfoil) 
    surf_uv = np.zeros((airfoil_pts.shape[0], 2))  # no-slip 
    # combine 
    xy_bnd = np.vstack([inlet_xy, far_xy, airfoil_pts]) 
    uv_bnd = np.vstack([inlet_uv, far_uv, surf_uv]) 
    # boundary type labels 
    bnd_types = np.array(['inlet']*inlet_xy.shape[0] + 
['far']*far_xy.shape[0] + ['wall']*airfoil_pts.shape[0]) 
    return xy_bnd, uv_bnd, bnd_types, airfoil_pts 
 
def sample_collocation(airfoil_pts, N_col_local=N_col, 
N_refine_local=N_refine): 
    # global uniform 
    x_all = x_min + (x_max - x_min)*np.random.rand(N_col_local,1) 
    y_all = y_min + (y_max - y_min)*np.random.rand(N_col_local,1) 
    # refine near airfoil box 
    ref_xmin, ref_xmax = center_x - 0.7*chord, center_x + 0.7*chord 
    ref_ymin, ref_ymax = center_y - 0.4*chord, center_y + 0.4*chord 
    xr = ref_xmin + (ref_xmax
ref_xmin)*np.random.rand(N_refine_local,1) 
    yr = ref_ymin + (ref_ymax
ref_ymin)*np.random.rand(N_refine_local,1) 
    xy_raw = np.vstack([np.hstack([x_all,y_all]), 
np.hstack([xr,yr])]) 
    mask_inside = is_inside_airfoil(xy_raw, airfoil_pts) 
    xy_col = xy_raw[~mask_inside] 
    return xy_col 
 
# ---------------- neural network ---------------- 
class DNN(nn.Module): 
    def __init__(self, in_dim=2, out_dim=3, hidden=100, layers=6, 
act=nn.Tanh()): 
        super().__init__() 
        layers_list = [] 
        layers_list.append(nn.Linear(in_dim, hidden)) 
        layers_list.append(act) 
        for _ in range(layers-1): 
            layers_list.append(nn.Linear(hidden, hidden)) 
            layers_list.append(act) 
        layers_list.append(nn.Linear(hidden, out_dim)) 
        self.net = nn.Sequential(*layers_list) 
        for m in self.net: 
            if isinstance(m, nn.Linear): 
                nn.init.xavier_uniform_(m.weight); 
nn.init.zeros_(m.bias) 
    def forward(self, x): 
        return self.net(x) 
 
# ---------------- PINN for steady incompressible NS ---------------- 
class PINN_NS: 
    def __init__(self, ub, lb, nu, device): 
        self.nu = nu 
        self.device = device 
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device) 
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device) 
        # network: increase capacity 
        self.model = DNN(in_dim=2, out_dim=3, hidden=100, 
layers=6).to(device)  # outputs: u, v, p 
        # optimizers & scheduler 
        self.adam = torch.optim.Adam(self.model.parameters(), lr=1e
3) 
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.adam, 
step_size=2000, gamma=0.7) 
        self.lbfgs_base = torch.optim.LBFGS(self.model.parameters(), 
lr=0.5, max_iter=100, history_size=50, line_search_fn='strong_wolfe') 
        # training helpers 
        self.loss_history = {'bc':[], 'pde':[], 'total':[]} 
        self.grad_clip = 5.0 
        # EMA for adaptive weighting 
        self.ema_pde = None 
        self.ema_bc = None 
        self.ema_alpha = 0.02 
        # pde minibatch size 
        self.pde_batch_size = 8000  # tune for your memory (reduce if 
OOM) 
 
    def predict(self, xy): 
        # xy: torch tensor (N,2) raw coordinates; normalize to [-1,1] 
        xy_norm = 2.0 * (xy - self.lb) / (self.ub - self.lb) - 1.0 
        out = self.model(xy_norm) 
        u = out[:,0:1]; v = out[:,1:2]; p = out[:,2:3] 
        return u, v, p 
 
    def pde_residuals(self, xy): 
        # xy: torch tensor (N,2) with requires_grad True 
        xy = xy.clone().detach().requires_grad_(True) 
        u, v, p = self.predict(xy) 
        # first derivatives 
        grads_u = grad(u.sum(), xy, create_graph=True)[0] 
        u_x = grads_u[:,0:1]; u_y = grads_u[:,1:2] 
        grads_v = grad(v.sum(), xy, create_graph=True)[0] 
        v_x = grads_v[:,0:1]; v_y = grads_v[:,1:2] 
        grads_p = grad(p.sum(), xy, create_graph=True)[0] 
        p_x = grads_p[:,0:1]; p_y = grads_p[:,1:2] 
        # second derivatives 
        u_xx = grad(u_x.sum(), xy, create_graph=True)[0][:,0:1] 
        u_yy = grad(u_y.sum(), xy, create_graph=True)[0][:,1:2] 
        v_xx = grad(v_x.sum(), xy, create_graph=True)[0][:,0:1] 
        v_yy = grad(v_y.sum(), xy, create_graph=True)[0][:,1:2] 
        # residuals 
        cont = u_x + v_y 
        mom_x = u*u_x + v*u_y + p_x - self.nu*(u_xx + u_yy) 
        mom_y = u*v_x + v*v_y + p_y - self.nu*(v_xx + v_yy) 
        return cont, mom_x, mom_y 
 
    def loss_bc(self, xy_bnd, uv_bnd, bnd_types, xy_outlet): 
        loss = 0.0 
        xy_bnd = torch.tensor(xy_bnd, 
dtype=torch.float32).to(self.device) 
        uv_bnd = torch.tensor(uv_bnd, 
dtype=torch.float32).to(self.device) 
        bnd_types = np.array(bnd_types) 
        # walls: u=v=0 
        mask_wall = (bnd_types == 'wall') 
        if mask_wall.sum() > 0: 
            xy_wall = xy_bnd[mask_wall] 
            u_pred, v_pred, _ = self.predict(xy_wall) 
            loss += torch.mean(u_pred**2 + v_pred**2) 
        # inlet and far: velocity components 
        mask_inlet = (bnd_types == 'inlet') 
        mask_far = (bnd_types == 'far') 
        if mask_inlet.sum() > 0: 
            xy_in = xy_bnd[mask_inlet] 
            uv_t = uv_bnd[mask_inlet] 
            u_pred, v_pred, _ = self.predict(xy_in) 
            loss += torch.mean((u_pred - uv_t[:,0:1])**2 + (v_pred - 
uv_t[:,1:2])**2) 
        if mask_far.sum() > 0: 
            xy_far = xy_bnd[mask_far] 
            uv_t = uv_bnd[mask_far] 
            u_pred, v_pred, _ = self.predict(xy_far) 
            loss += torch.mean((u_pred - uv_t[:,0:1])**2 + (v_pred - 
uv_t[:,1:2])**2) 
        # outlet: pressure ~ 0 
        if xy_outlet is not None and xy_outlet.shape[0]>0: 
            xy_out = torch.tensor(xy_outlet, 
dtype=torch.float32).to(self.device) 
            _, _, p_pred = self.predict(xy_out) 
            loss += torch.mean(p_pred**2) 
        return loss 
 
    def loss_pde(self, xy_col): 
        xy_col = torch.tensor(xy_col, 
dtype=torch.float32).to(self.device) 
        cont, mx, my = self.pde_residuals(xy_col) 
        mse = torch.mean(cont**2) + torch.mean(mx**2) + 
torch.mean(my**2) 
        return mse 
 
    # ---------------- Hybrid training (Adam minibatch + LBFGS on 
full set) ---------------- 
    def train_hybrid(self, xy_col, xy_bnd, uv_bnd, bnd_types, 
xy_outlet, 
                     n_adam_pre=8000, n_cycles=3, n_adam_cycle=3000, 
lbfgs_iter_per_cycle=500, 
                     w_bc_initial=10.0, use_lbfgs_final=True): 
        # convert arrays 
        xy_col = np.array(xy_col) 
        N_col_total = xy_col.shape[0] 
        xy_col_t_full = torch.tensor(xy_col, 
dtype=torch.float32).to(self.device) 
        xy_bnd_t = torch.tensor(xy_bnd, 
dtype=torch.float32).to(self.device) 
        uv_bnd_t = torch.tensor(uv_bnd, 
dtype=torch.float32).to(self.device) 
        bnd_types_arr = np.array(bnd_types) 
        xy_out_t = torch.tensor(xy_outlet, 
dtype=torch.float32).to(self.device) if xy_outlet is not None else 
None 
 
        # Pretrain with Adam (minibatches) 
        print(f"Pretraining Adam: {n_adam_pre} iterations (batch size 
{self.pde_batch_size})") 
        for it in range(n_adam_pre): 
            idx = np.random.choice(N_col_total, 
size=min(self.pde_batch_size, N_col_total), replace=False) 
            xy_col_batch = torch.tensor(xy_col[idx], 
dtype=torch.float32).to(self.device) 
 
            self.adam.zero_grad() 
            loss_bc = self.loss_bc(xy_bnd_t, uv_bnd_t, bnd_types_arr, 
xy_out_t) 
            loss_pde = self.loss_pde(xy_col_batch) 
 
            # EMA updates 
            with torch.no_grad(): 
                lp = loss_pde.item(); lb = loss_bc.item() 
                if self.ema_pde is None: 
                    self.ema_pde = lp; self.ema_bc = lb 
                else: 
                    self.ema_pde = (1 - self.ema_alpha) * 
self.ema_pde + self.ema_alpha * lp 
                    self.ema_bc = (1 - self.ema_alpha) * self.ema_bc 
+ self.ema_alpha * lb 
            ratio = (self.ema_pde / (self.ema_bc + 1e-12)) 
            w_bc = float(np.clip(w_bc_initial * ratio, 2.0, 50)) 
 
            loss = w_bc * loss_bc + loss_pde 
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
max_norm=self.grad_clip) 
            self.adam.step() 
            self.scheduler.step() 
 
            if it % 100 == 0 or it == n_adam_pre - 1: 
                print(f"[Pre-Adam] it={it:05d} 
Total={loss.item():.3e} BC={loss_bc.item():.3e} 
PDE={loss_pde.item():.3e} w_bc={w_bc:.2f}") 
                self.loss_history['bc'].append(loss_bc.item()); 
self.loss_history['pde'].append(loss_pde.item()); 
self.loss_history['total'].append(loss.item()) 
 
        # cycles of Adam bursts + LBFGS 
        for cyc in range(n_cycles): 
            print(f"\n=== Cycle {cyc+1}/{n_cycles}: Adam 
{n_adam_cycle} iters then LBFGS {lbfgs_iter_per_cycle} iters ===") 
            for it in range(n_adam_cycle): 
                idx = np.random.choice(N_col_total, 
size=min(self.pde_batch_size, N_col_total), replace=False) 
                xy_col_batch = torch.tensor(xy_col[idx], 
dtype=torch.float32).to(self.device) 
 
                self.adam.zero_grad() 
                loss_bc = self.loss_bc(xy_bnd_t, uv_bnd_t, 
bnd_types_arr, xy_out_t) 
                loss_pde = self.loss_pde(xy_col_batch) 
 
                with torch.no_grad(): 
                    self.ema_pde = (1 - self.ema_alpha) * 
self.ema_pde + self.ema_alpha * loss_pde.item() 
                    self.ema_bc = (1 - self.ema_alpha) * self.ema_bc 
+ self.ema_alpha * loss_bc.item() 
                ratio = (self.ema_pde / (self.ema_bc + 1e-12)) 
                w_bc = float(np.clip(w_bc_initial * ratio * (0.9 ** 
(cyc)), 2.0, 50)) 
 
                loss = w_bc * loss_bc + loss_pde 
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(
), max_norm=self.grad_clip) 
                self.adam.step() 
 
                if it % 200 == 0 or it == n_adam_cycle - 1: 
                    print(f"[Cycle{cyc+1}-Adam] it={it:04d} 
Total={loss.item():.3e} BC={loss_bc.item():.3e} 
PDE={loss_pde.item():.3e} w_bc={w_bc:.2f}") 
            # LBFGS refinement on full collocation 
            def closure_lbfgs(): 
                self.lbfgs_base.zero_grad() 
                loss_bc_loc = self.loss_bc(xy_bnd_t, uv_bnd_t, 
bnd_types_arr, xy_out_t) 
                loss_pde_loc = self.loss_pde(xy_col_t_full) 
                loss_loc = (w_bc_initial * 0.5 * (0.9 ** cyc)) * 
loss_bc_loc + loss_pde_loc 
                loss_loc.backward() 
                return loss_loc 
 
            print("Starting LBFGS refinement (full collocation) ...") 
            lbfgs_tmp = torch.optim.LBFGS(self.model.parameters(), 
lr=0.1, max_iter=lbfgs_iter_per_cycle, history_size=50, 
line_search_fn='strong_wolfe') 
            lbfgs_tmp.step(closure_lbfgs) 
            print("LBFGS cycle finished.") 
 
        # final LBFGS polishing 
        if use_lbfgs_final: 
            print("\nFinal LBFGS polishing ...") 
            def closure_final(): 
                self.lbfgs_base.zero_grad() 
                loss_bc_f = self.loss_bc(xy_bnd_t, uv_bnd_t, 
bnd_types_arr, xy_out_t) 
                loss_pde_f = self.loss_pde(xy_col_t_full) 
                loss_f = (w_bc_initial * 0.4) * loss_bc_f + 
loss_pde_f 
                loss_f.backward() 
                return loss_f 
            lbfgs_final = torch.optim.LBFGS(self.model.parameters(), 
lr=0.1, max_iter=1000, history_size=200, 
line_search_fn='strong_wolfe') 
            lbfgs_final.step(closure_final) 
            print("Final LBFGS done.") 
 
        print("Hybrid training finished.") 
 
    # ---------------- prediction / utility methods ---------------- 
    def predict_grid(self, X, Y, airfoil_pts): 
        xy = np.vstack([X.ravel(), Y.ravel()]).T 
        mask_inside = is_inside_airfoil(xy, airfoil_pts) 
        xy_t = torch.tensor(xy, dtype=torch.float32).to(self.device) 
        with torch.no_grad(): 
            u_pred, v_pred, p_pred = self.predict(xy_t) 
            u = u_pred.cpu().numpy().reshape(-1) 
            v = v_pred.cpu().numpy().reshape(-1) 
            p = p_pred.cpu().numpy().reshape(-1) 
        u_masked = np.where(~mask_inside, u, np.nan).reshape(X.shape) 
        v_masked = np.where(~mask_inside, v, np.nan).reshape(X.shape) 
        p_masked = np.where(~mask_inside, p, np.nan).reshape(X.shape) 
        Vmag = np.sqrt(np.nan_to_num(u_masked)**2 + 
np.nan_to_num(v_masked)**2) 
        return u_masked, v_masked, p_masked, Vmag 
 
    def predict_on_surface(self, surface_pts): 
        xy = torch.tensor(surface_pts, 
dtype=torch.float32).to(self.device) 
        with torch.no_grad(): 
            u_pred, v_pred, p_pred = self.predict(xy) 
        return u_pred.cpu().numpy().reshape(-1), 
v_pred.cpu().numpy().reshape(-1), p_pred.cpu().numpy().reshape(-1) 
 
    def compute_forces(self, surface_pts): 
        pts = surface_pts 
        ds = [] 
        nx = []; ny = [] 
        for i in range(len(pts)): 
            p0 = pts[i] 
            p1 = pts[(i+1)%len(pts)] 
            dx = p1[0]-p0[0]; dy = p1[1]-p0[1] 
            s = np.sqrt(dx*dx + dy*dy) 
            ds.append(s) 
            n_x = dy / (s+1e-12) 
            n_y = -dx / (s+1e-12) 
            nx.append(n_x); ny.append(n_y) 
        ds = np.array(ds); nx = np.array(nx); ny = np.array(ny) 
        _, _, p_vals = self.predict_on_surface(pts) 
        Lp = -np.sum(p_vals * ny * ds) 
        Dp = -np.sum(p_vals * nx * ds) 
        return Lp, Dp, p_vals 
 
# ---------------- run a case ---------------- 
def run_case(alpha_deg, save_prefix="case", hybrid_params=None, 
quick_test=False): 
    print(f"\n=== Running AoA = {alpha_deg} deg ===") 
    xy_bnd, uv_bnd, bnd_types, airfoil_pts = 
sample_boundary_points(alpha_deg) 
    # for quick tests reduce collocation counts 
    if quick_test: 
        xy_col = sample_collocation(airfoil_pts, N_col_local=3000, 
N_refine_local=800) 
    else: 
        xy_col = sample_collocation(airfoil_pts, N_col_local=N_col, 
N_refine_local=N_refine) 
    # outlet points (x == x_max) 
    out_idx = np.where(np.isclose(xy_bnd[:,0], x_max))[0] 
    xy_outlet = xy_bnd[out_idx] if out_idx.size>0 else None 
 
    pinn = PINN_NS(ub=ub, lb=lb, nu=nu, device=device) 
 
    # hybrid training params default 
    if hybrid_params is None: 
        hybrid_params = dict(n_adam_pre=8000, n_cycles=3, 
n_adam_cycle=4000, 
                             lbfgs_iter_per_cycle=1000, 
w_bc_initial=50.0, use_lbfgs_final=True) 
 
    t0 = time.time() 
    pinn.train_hybrid(xy_col, xy_bnd, uv_bnd, bnd_types, xy_outlet, 
**hybrid_params) 
    elapsed = time.time()-t0 
    print(f"Training finished in {elapsed:.1f} s ({elapsed/60:.2f} 
min)") 
 
    # evaluate on grid 
    xs = np.arange(x_min, x_max, 0.0005) 
    ys = np.arange(y_min, y_max, 0.0005) 
    X, Y = np.meshgrid(xs, ys) 
    u, v, p, V = pinn.predict_grid(X, Y, airfoil_pts) 
 
    # compute forces approx 
    Lp, Dp, p_surf = pinn.compute_forces(airfoil_pts) 
    print(f"Approx lift per unit span (non-dim): {Lp:.5f}, drag per 
unit span (non-dim): {Dp:.5f}") 
 
    # plotting 
    fig, axes = plt.subplots(2,2, figsize=(12,10)) 
    ax = axes[0,0] 
    im = ax.imshow(p, extent=[x_min, x_max, y_min, y_max], 
origin='lower', cmap='RdBu_r') 
    ax.plot(airfoil_pts[:,0], airfoil_pts[:,1], 'k-', linewidth=1.5) 
    ax.set_title(f'Pressure (AoA={alpha_deg}°)') 
    plt.colorbar(im, ax=ax) 
    ax = axes[0,1] 
    im2 = ax.imshow(V, extent=[x_min, x_max, y_min, y_max], 
origin='lower', cmap='jet') 
    ax.plot(airfoil_pts[:,0], airfoil_pts[:,1], 'k-', linewidth=1.5) 
    ax.set_title(f'Velocity magnitude (AoA={alpha_deg}°)') 
    plt.colorbar(im2, ax=ax) 
 
    ax = axes[1,0] 
    # streamplot expects 2D arrays without NaN; we supply u,v with 
NaNs -> streamplot ignores NaN but may warn. 
    ax.streamplot(X, Y, np.nan_to_num(u), np.nan_to_num(v), 
density=1.2, color='k', linewidth=0.5) 
    ax.plot(airfoil_pts[:,0], airfoil_pts[:,1], 'k-', linewidth=1.5) 
    ax.set_title('Streamlines') 
 
    ax = axes[1,1] 
    cp_vals = p_surf 
    ax.plot(np.linspace(0,1,len(cp_vals)), cp_vals, '-o', 
markersize=2) 
    ax.set_title('Surface pressure (approx sample)') 
    ax.set_xlabel('surface index ~ x/c') 
    ax.set_ylabel('p (non-dim)') 
 
    plt.suptitle(f"NACA6409 PINN results AoA={alpha_deg} deg") 
    plt.tight_layout(rect=[0,0,1,0.96]) 
    out_png = f"{save_prefix}_AoA_{alpha_deg}.png" 
    fig.savefig(out_png, dpi=300) 
    print("Saved figure:", out_png) 
 
    return pinn, X, Y, u, v, p, V, airfoil_pts 
 
# ---------------- main ---------------- 
if __name__ == "__main__": 
    # quick_test=True for fast debug (faster runs, less accurate) 
    hybrid_params = dict(n_adam_pre=8000, n_cycles=3, 
n_adam_cycle=4000, 
                         lbfgs_iter_per_cycle=500, w_bc_initial=50, 
use_lbfgs_final=True) 
 
    # run AoA=0 then AoA=5 
    for aoa in [0]: 
        run_case(aoa, save_prefix="NACA6409_PINN", 
hybrid_params=hybrid_params, quick_test=True)
