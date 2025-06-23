# main.py
from test_cases import ode_cases
from PINN import PINN, train
import torch

gen_testdata, gen_bc, pde_residual = ode_cases[1]
model = PINN([1, 20, 20, 1]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model, x_test, u_test = train(
    model, optimizer,
    gen_testdata, gen_bc, pde_residual,
    epochs=3000, refine_every=1000, relaxation=1.0
)

