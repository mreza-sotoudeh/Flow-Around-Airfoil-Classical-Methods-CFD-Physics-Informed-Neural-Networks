import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
from torch.autograd import grad 
from matplotlib.path import Path 
  --- تنظیمات سیستم  --- 
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
dtype = torch.float32 
  --- پارامترهای فیزیکی --- 
#
Re = 100.0 
nu = 1.0 / Re 
alpha_deg = 5.0 
alpha_rad = np.radians(alpha_deg) 
U_inf = 1.0 
u_far = U_inf * np.cos(alpha_rad) 
v_far = U_inf * np.sin(alpha_rad) 
  ابعاد دامنه  
#
x_min, x_max = -0.5, 1.5 
y_min, y_max = -0.7, 0.7 
  --- توابع کمکی --- 
#
def get_naca0024_pts(n_points=1000): 
    beta = np.linspace(0, np.pi, n_points) 
    x = 0.5 * (1 - np.cos(beta)) 
    yt = 5 * 0.24 * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 
0.2843*x**3 - 0.1015*x**4) 
    upper = np.stack([x, yt], axis=1) 
    lower = np.stack([x[::-1], -yt[::-1]], axis=1) 
    return np.vstack([upper, lower]).astype(np.float32) 
  --- معماری شبکه عصبی  --- 
#
class DeepPINN(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.net = nn.Sequential( 
            nn.Linear(2, 128), nn.Tanh(), 
            nn.Linear(128, 128), nn.Tanh(), 
            nn.Linear(128, 128), nn.Tanh(), 
            nn.Linear(128, 128), nn.Tanh(), 
            nn.Linear(128, 3) # u, v, p 
        ) 
        for m in self.net: 
            if isinstance(m, nn.Linear): 
                nn.init.xavier_normal_(m.weight) 
 
    def forward(self, x):  سازی داخلی مختصات نرمال  #       
 
        x_norm = 2.0 * (x[:, 0:1] - x_min) / (x_max - x_min) - 1.0 
        y_norm = 2.0 * (x[:, 1:2] - y_min) / (y_max - y_min) - 1.0 
        return self.net(torch.cat([x_norm, y_norm], dim=1)) 
  --- گر کالس اصلی حل  --- 
#
class Solver: 
    def __init__(self): 
        self.model = DeepPINN().to(device) 
        self.airfoil_pts = torch.tensor(get_naca0024_pts(), 
device=device, dtype=dtype) 
  تولید نقاط کلوکیشن  #       
 
        pts_global = np.random.uniform([x_min, y_min], [x_max, 
y_max], (12000, 2)) 
        pts_near = np.random.uniform([-0.1, -0.3], [1.1, 0.3], (6000, 
2)) 
        xy_all = np.vstack([pts_global, pts_near]).astype(np.float32) 
        mask = Path(self.airfoil_pts.cpu()).contains_points(xy_all) 
        self.xy_col = torch.tensor(xy_all[~mask], device=device, 
dtype=dtype, requires_grad=True) 
  نقاط مرزی برای فشار و سرعت دوردست  #       
 
        self.xy_far = torch.tensor(np.random.uniform([x_min, y_min], 
[x_max, y_max], (1500, 2)), device=device, dtype=dtype) 
        self.xy_p_ref = torch.tensor([[x_max, 0.0]], device=device, 
dtype=dtype) 
 
    def predict(self, xy): 
        out = self.model(xy) 
        return out[:, 0:1], out[:, 1:2], out[:, 2:3] 
 
    def calc_loss(self, it): 
        u, v, p = self.predict(self.xy_col) 
 
  محاسبه مشتقات  #       
 
        u_g = grad(u.sum(), self.xy_col, create_graph=True)[0]; u_x, 
u_y = u_g[:,0:1], u_g[:,1:2] 
        v_g = grad(v.sum(), self.xy_col, create_graph=True)[0]; v_x, 
v_y = v_g[:,0:1], v_g[:,1:2] 
        p_g = grad(p.sum(), self.xy_col, create_graph=True)[0]; p_x, 
p_y = p_g[:,0:1], p_g[:,1:2] 
 
        u_xx = grad(u_x.sum(), self.xy_col, 
create_graph=True)[0][:,0:1] 
        u_yy = grad(u_y.sum(), self.xy_col, 
create_graph=True)[0][:,1:2] 
        v_xx = grad(v_x.sum(), self.xy_col, 
create_graph=True)[0][:,0:1] 
        v_yy = grad(v_y.sum(), self.xy_col, 
create_graph=True)[0][:,1:2] 
  )PDE( معادالت فیزیکی #       
 
        loss_pde = torch.mean((u_x + v_y)**2) + \ 
                   torch.mean((u*u_x + v*u_y + p_x - nu*(u_xx + 
u_yy))**2) + \ 
                   torch.mean((u*v_x + v*v_y + p_y - nu*(v_xx + 
v_yy))**2) 
  )BC( شرایط مرزی #       
 
        u_s, v_s, _ = self.predict(self.airfoil_pts) 
        loss_wall = torch.mean(u_s**2 + v_s**2) 
 
        u_f, v_f, _ = self.predict(self.xy_far) 
        loss_far = torch.mean((u_f - u_far)**2 + (v_f - v_far)**2) 
 
        _, _, p_ref = self.predict(self.xy_p_ref) 
        loss_p = torch.mean(p_ref**2) 
 
        w_bc = 5.0 if it > 5000 else 5.0 
        return loss_pde, (loss_wall + loss_far + loss_p) * w_bc 
 
    def train(self, epochs=5000): 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e
3) 
        print("Training starting...") 
        for it in range(epochs + 1): 
            optimizer.zero_grad() 
            lpde, lbc = self.calc_loss(it) 
            loss = lpde + lbc 
            loss.backward() 
            optimizer.step() 
 
            if it % 500 == 0: 
                print(f"it={it:05d} Total={loss.item():.3e} 
BC={lbc.item():.3e} PDE={lpde.item():.3e}") 
 
        print("\nSwitching to L-BFGS for refinement...") 
        lbfgs = torch.optim.LBFGS(self.model.parameters(), 
max_iter=1000, line_search_fn="strong_wolfe") 
        def closure(): 
            lbfgs.zero_grad() 
            lpde, lbc = self.calc_loss(epochs) 
            (lpde + lbc).backward() 
            return lpde + lbc 
        lbfgs.step(closure) 
 
    def plot(self): 
        res = 150 
        x = np.linspace(x_min, x_max, res) 
        y = np.linspace(-0.5, 0.5, res) 
        X, Y = np.meshgrid(x, y) 
        xy_t = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), 
device=device, dtype=dtype) 
 
        with torch.no_grad(): 
            u, v, p = self.predict(xy_t) 
            U = u.cpu().numpy().reshape(res, res) 
            V = v.cpu().numpy().reshape(res, res) 
            P = p.cpu().numpy().reshape(res, res) 
            Vel = np.sqrt(U**2 + V**2) 
 
        mask = 
Path(self.airfoil_pts.cpu()).contains_points(xy_t.cpu().numpy()).resh
ape(res, res) 
        U[mask] = V[mask] = P[mask] = Vel[mask] = np.nan 
 
        fig, ax = plt.subplots(1, 3, figsize=(22, 6)) 
  Colorbar + کانتور فشار #       
 
        cp = ax[0].contourf(X, Y, P, levels=80, cmap='RdBu_r') 
        ax[0].set_title("Pressure Field") 
        fig.colorbar(cp, ax=ax[0], fraction=0.046, pad=0.04) 
  Colorbar + کانتور سرعت #       
        cv = ax[1].contourf(X, Y, Vel, levels=80, cmap='jet') 
        ax[1].set_title("Velocity Magnitude") 
        fig.colorbar(cv, ax=ax[1], fraction=0.046, pad=0.04) 
  خطوط جریان #       
 
        ax[2].streamplot(X, Y, np.nan_to_num(U), np.nan_to_num(V), 
color='k', density=1.5) 
        ax[2].set_title("Streamlines") 
 
        for a in ax: 
            a.fill(self.airfoil_pts.cpu()[:,0], 
self.airfoil_pts.cpu()[:,1], 'k', zorder=10) 
            a.set_aspect('equal') 
            a.set_xlim(-0.3, 1.3) 
 
        plt.tight_layout() 
        plt.show() 
  --- اجرا --- 
#
if __name__ == "__main__": 
    solver = Solver() 
    solver.train(epochs=5000) #  برای دقت باالتر اپوک را بیشتر کنید 
    solver.plot() 
