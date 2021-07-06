#!/bin/bash

conda create -n raylearn -c pytorch pytorch
conda activate raylearn
conda install scipy matplotlib h5py sympy tqdm imageio spyder
