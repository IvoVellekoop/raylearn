=== Installing the required packages ===

1a. Install Anaconda or Miniconda either from the website www.anaconda.com,

1b. or install using your package manager. On Windows, you can use winget to install it:
winget install Anaconda.Anaconda3

Note: you may need to reopen your terminal before the conda command is available. On Windows, you can use either the Anaconda Cmd or Powershell, or regular Cmd or Powershell. On Linux, see step 2.


2. On Windows, skip this step. On Linux, you may need to initialize conda by running:
source [PATH TO CONDA]/bin/activate

And then run:
conda init

This adds some lines to your ~/.bashrc, and activates the base environment by default. You can disable this by running:
conda config --set auto_activate_base false


3. More details on installation can be found here: https://docs.anaconda.com/anaconda/install/


4a. Install the exact package versions using the environment snapshot yml file. Navigate to the raylearn folder in your terminal, and the command suitable for your platform (or alternatively do a fresh install, see step 4b).

conda env create -f environment-windows.yml
or
conda env create -f environment-linux.yml

and then activate it:
conda activate raylearn

4b. Or perform a fresh install by running these commands

conda create -n raylearn -c pytorch pytorch
conda activate raylearn
conda install scipy numpy matplotlib h5py sympy tqdm imageio pytest flake8 autopep8 pyopengl
conda install -c conda-forge pyglfw hdf5storage

