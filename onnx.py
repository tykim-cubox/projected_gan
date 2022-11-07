import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy


network_pkl = '/home/aiteam/tykim/DE-GAN/projected_gan/pokemon.pkl'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
with dnnlib.util.open_url(network_pkl) as f:
  G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

seed = 30
noise_mode = 'const'
z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device).float()
truncation_psi = 1.0
label = torch.zeros([1, G.c_dim], device=device)

# G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)

torch.onnx.export(G, args=(z, label, truncation_psi, noise_mode),
                  f='./generator.onnx', verbose=True, input_names=['z','label','truncation_psi','noise_mode'], output_names=['images'], opset_version=12) 
                  # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
