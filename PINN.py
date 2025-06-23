# PINN.py
import torch
import torch.nn as nn
import numpy as np
from adaptive import node_moving_1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.act = torch.tanh
        self.layers = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1])
            for i in range(len(layers) - 1)
        ])
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)

def train(model, optimizer, gen_testdata, gen_bc, pde_residual,
          n_dom=200, n_bc=200, n_test=1000, epochs=1000,
          refine_every=None, relaxation=1.0):
    x_dom = np.random.uniform(0, 1, (n_dom, 1))
    x_bc, u_bc = gen_bc(n_bc)
    x_test, u_test = gen_testdata(n_test)
    x_bc_t = torch.tensor(x_bc, dtype=torch.float32, device=device)
    u_bc_t = torch.tensor(u_bc, dtype=torch.float32, device=device)

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        x_dom_t = torch.tensor(x_dom, dtype=torch.float32, device=device)
        loss_pde = (pde_residual(model, x_dom_t)**2).mean()
        u_pred_bc = model(x_bc_t)
        loss_bc = ((u_pred_bc - u_bc_t)**2).mean()
        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()

        if refine_every and ep % refine_every == 0:
            with torch.no_grad():
                r = pde_residual(model, x_dom_t).abs().cpu().numpy().flatten()
                bc_err = np.zeros_like(r)
                bc_err[0] = abs(model(torch.tensor([[0.0]], dtype=torch.float32, device=device)).item() - u_bc[0])
                bc_err[-1] = abs(model(torch.tensor([[1.0]], dtype=torch.float32, device=device)).item() - u_bc[-1])
                total_err = r + bc_err
                x_dom = node_moving_1d(
                    x_dom.flatten(),
                    total_err,
                    boundary_nodes_id=np.array([0, len(x_dom)-1]),
                    ratio_nodal_distance=5,
                    relaxation=relaxation
                ).reshape(-1, 1)
    with torch.no_grad():
        u_pred = model(torch.tensor(x_test, dtype=torch.float32, device=device))
        l2 = float(np.linalg.norm(u_pred.cpu().numpy() - u_test) / np.linalg.norm(u_test))
        print(f"FINISHED: Epoch {epochs}:\t L2 error = {l2:.6f}")

    return model, x_test, u_test
