# test_cases.py
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Case 1: Single cosine wave
# -------------------------
def gen_testdata_1(n=1000):
    x = np.linspace(0, 1, n)[:, None]
    u = 10 * np.cos(np.pi * (x - 0.5) / 2)
    return x, u

def gen_bc_1(n=200):
    x = np.vstack([np.zeros((n // 2, 1)), np.ones((n - n // 2, 1))])
    u = 10 * np.cos(np.pi * (x - 0.5) / 2)
    return x, u

def pde_residual_1(model, x_tensor):
    x = x_tensor.clone().detach().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    rhs = -5 * np.pi * torch.sin(np.pi * (x - 0.5) / 2)
    return u_x - rhs


# -------------------------
# Case 2: Steep V-shape (smooth)
# -------------------------
def gen_testdata_2(n=1000):
    k, x0 = 200.0, 0.5
    x = np.linspace(0, 1, n)[:, None]
    C0 = np.log(np.cosh(-k * x0)) / k
    u = (np.log(np.cosh(k * (x - x0))) / k) - C0
    return x, u

def gen_bc_2(n=200):
    k, x0 = 200.0, 0.5
    x = np.vstack([np.zeros((n // 2, 1)), np.ones((n - n // 2, 1))])
    C0 = np.log(np.cosh(-k * x0)) / k
    u = (np.log(np.cosh(k * (x - x0))) / k) - C0
    return x, u

def pde_residual_2(model, x_tensor):
    k, x0 = 200.0, 0.5
    x = x_tensor.clone().detach().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    rhs = torch.tanh(k * (x - x0))
    return u_x - rhs


# -------------------------
# Case 3: Steep tanh ramp
# -------------------------
def gen_testdata_3(n=1000):
    k, x0 = 150.0, 0.6
    x = np.linspace(0, 1, n)[:, None]
    C0 = 0.5 * (1 + np.tanh(-k * x0))
    u = 0.5 * (1 + np.tanh(k * (x - x0))) - C0
    return x, u

def gen_bc_3(n=200):
    k, x0 = 150.0, 0.6
    x = np.vstack([np.zeros((n // 2, 1)), np.ones((n - n // 2, 1))])
    C0 = 0.5 * (1 + np.tanh(-k * x0))
    u = 0.5 * (1 + np.tanh(k * (x - x0))) - C0
    return x, u

def pde_residual_3(model, x_tensor):
    k, x0 = 150.0, 0.6
    x = x_tensor.clone().detach().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    rhs = 0.5 * k * (1 / torch.cosh(k * (x - x0))**2)
    return u_x - rhs


# -------------------------
# Case 4: Exponential increase
# -------------------------
def gen_testdata_4(n=1000):
    u0, u1, k = 0.0, 5.0, 10.0
    x = np.linspace(0, 1, n)[:, None]
    u = u0 + (u1 - u0) * (1 - np.exp(k * x)) / (1 - np.exp(k))
    return x, u

def gen_bc_4(n=200):
    u0, u1, k = 0.0, 5.0, 10.0
    x = np.vstack([np.zeros((n // 2, 1)), np.ones((n - n // 2, 1))])
    u = u0 + (u1 - u0) * (1 - np.exp(k * x)) / (1 - np.exp(k))
    return x, u

def pde_residual_4(model, x_tensor):
    u0, u1, k = 0.0, 5.0, 10.0
    L = 1.0
    x = x_tensor.clone().detach().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    rhs = (u1 - u0) * (-k * torch.exp(k * x)) / (1 - torch.exp(torch.tensor(k * L)))
    return u_x - rhs


# -------------------------
# Case 5: Multi-frequency decaying cosine
# -------------------------
def gen_testdata_5(n=1000):
    x = np.linspace(0, 1, n)[:, None]
    u = np.exp(-x) * np.cos(5 * np.pi * (x - 0.5))
    return x, u

def gen_bc_5(n=200):
    x = np.vstack([np.zeros((n // 2, 1)), np.ones((n - n // 2, 1))])
    u = np.exp(-x) * np.cos(5 * np.pi * (x - 0.5))
    return x, u

def pde_residual_5(model, x_tensor):
    x = x_tensor.clone().detach().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    expm = torch.exp(-x)
    arg = 5 * np.pi * (x - 0.5)
    rhs = -expm * torch.cos(arg) + expm * 5 * np.pi * torch.sin(arg)
    return u_x - rhs


# -------------------------
# Case 6: Piecewise constant (discontinuous)
# -------------------------
def gen_testdata_6(n=1000):
    x = np.linspace(0, 1, n)[:, None]
    u = np.where(x < 0.5, 10.0, 1.0)
    return x, u

def gen_bc_6(n=200):
    x = np.vstack([np.zeros((n // 2, 1)), np.ones((n - n // 2, 1))])
    u = np.where(x < 0.5, 10.0, 1.0)
    return x, u

def pde_residual_6(model, x_tensor):
    x = x_tensor.clone().detach().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return u_x


# -------------------------
# Case Dictionary
# -------------------------
ode_cases = {
    1: (gen_testdata_1, gen_bc_1, pde_residual_1),
    2: (gen_testdata_2, gen_bc_2, pde_residual_2),
    3: (gen_testdata_3, gen_bc_3, pde_residual_3),
    4: (gen_testdata_4, gen_bc_4, pde_residual_4),
    5: (gen_testdata_5, gen_bc_5, pde_residual_5),
    6: (gen_testdata_6, gen_bc_6, pde_residual_6)
}
