import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from scipy.linalg import solve
from scipy.stats import norm
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from scipy.interpolate import interp1d

from CodeShort.test_cases import ode_cases
from CodeShort.PINN import PINN, train
import torch


if __name__ == "__main__":
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    gen_testdata, gen_bc, pde_residual = ode_cases[1]
    model = PINN([1, 20, 20, 1]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model, x_test, u_test = train(
        model, optimizer,
        gen_testdata, gen_bc, pde_residual,
        epochs=3000, refine_every=1000, relaxation=1.0
    )

