import torch
from torch import tensor
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from plot_functions import plot_coords, format_prefix
from testing import MSE
from tpm import TPM
from vector_functions import components


matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-1x170um/raylearn_pencil_beam_738714.642102_1x170um.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-1x170um_2nd/raylearn_pencil_beam_738714.744282_1x170um_2nd.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-2x170um/raylearn_pencil_beam_738714.652131_2x170um.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-2x170um_2nd/raylearn_pencil_beam_738714.757416_2x170um_2nd.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-3x170um/raylearn_pencil_beam_738714.670969_3x170um.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-3x170um_2nd/raylearn_pencil_beam_738714.768351_3x170um_2nd.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-4x170um/raylearn_pencil_beam_738714.683205_4x170um.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-4x170um_2nd/raylearn_pencil_beam_738714.784681_4x170um_2nd.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-5x170um/raylearn_pencil_beam_738714.692862_5x170um.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-5x170um_2nd/raylearn_pencil_beam_738714.793049_5x170um_2nd.mat"
# matpath = "LocalData/raylearn-data/TPM/pencil-beam-positions/12-Jul-2022-1x400um/raylearn_pencil_beam_738714.801627_1x400um.mat"

matfile = h5py.File(matpath, 'r')

cam_ft_coords_gt = tensor((matfile['cam_ft_col'], matfile['cam_ft_row'])).permute(1, 2, 0)
cam_im_coords_gt = tensor((matfile['cam_img_col'], matfile['cam_img_row'])).permute(1, 2, 0)

tpm = TPM()
tpm.set_measurement(matfile)
tpm.update()

# Compute the magnification of coordinate 'spread'
slm_coord_spread_m = tpm.slm_coords.std(-2) * tpm.slm_height
tan_angle_at_sample_plane_spread = slm_coord_spread_m
cam_im_spread_from_slm_m = cam_im_coords_gt.std(dim=-2) * tpm.cam_pixel_size
cam_im_spread_per_slm_spread = cam_im_spread_from_slm_m  / slm_coord_spread_m

print(f'\nSLM to image cam magnification: {cam_im_spread_per_slm_spread.mean().detach().item():.3f}')