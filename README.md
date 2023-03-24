# RayLearn

RayLearn is a sequential ray tracer written in PyTorch, capable of both forward and inverse ray tracing. Forward ray tracing can be used to compute ray positions, directions, pathlengths, and phase. Inverse ray tracing can be used to fit multiple model parameters (e.g. the thickness of a slab of glass, or the focal distance of an Abbe sine lens) to a set of input/output rays.

## Installing the required Python packages from Anaconda

1a: Install Anaconda or Miniconda either from the website www.anaconda.com,

1b: or install using your package manager. On Windows, you can use winget to install it:
`winget install Anaconda.Anaconda3`

Note: you may need to reopen your terminal before the conda command is available. On Windows, you can use either the Anaconda Cmd or Powershell, or regular Cmd or Powershell. On Linux, see step 2.

2: On Windows, skip this step. On Linux, you may need to initialize conda by running:
`source [PATH TO CONDA]/bin/activate`

And then run:
`conda init`

This adds some lines to your ~/.bashrc, and activates the base environment by default. You can disable this by running:
`conda config --set auto_activate_base false`

3: More details on installation can be found here: https://docs.anaconda.com/anaconda/install/

4a: Install the exact package versions using the environment snapshot yml file. Navigate to the raylearn folder in your terminal, and the command suitable for your platform (or alternatively do a fresh install, see step 4b).
`conda env create -f environment-windows.yml`
or
`conda env create -f environment-linux.yml`

and then activate it:
`conda activate raylearn`

4b: Or perform a fresh install by running these commands

```
conda create -n raylearn -c pytorch pytorch
conda activate raylearn
conda install scipy numpy matplotlib h5py sympy tqdm imageio pytest flake8 autopep8 pyopengl
conda install -c conda-forge pyglfw hdf5storage
```

## Installing VS Code extensions for debugging Python

Install the VS Code flavour of your choosing: VS Code (Microsoft's License, partially open source vscode build) or VS Codium (MIT License, Fully open source vscode build) or Code OSS (MIT License, Fully open source vscode).

**VS Codium and Code OSS:**

1a: Download VS Codium from the website https://vscodium.com/

1b: or install with a package manager,
    - On Windows, e.g.: `winget install VSCodium.VSCodium`
    - On Linux, if it's not available through your distribution's package manager, there is the option to use Flatpak or Snap.

2: Install the extensions for debugging:

- Go to View → Extensions, or Sidebar → Extensions
- Install the *Anaconda Extension Pack* by ms-python
- Optionally install *Pyright* by ms-pyright for static type checking,
  or any type checker of your own choosing.

**VS Code:**

1a: Download from the website https://code.visualstudio.com/

1b: or install with a package manager, e.g. on Windows: `winget install Microsoft.VisualStudioCode`

2: Install the extensions for debugging:

- Go to View → Extensions, or the sidebar → Extensions
- Install the *Python* extension by Microsoft
- Optionally check if the *Pylance* extension (for static type checking) is
  automatically installed along with the *Python* extension. If not, you can
  install it manually. Or use any type checker of your own choosing.

## Getting started

Running/debugging the python files in VS Code:

1. Follow the steps in *Installing the required Python packages from Anaconda* and *Installing VS Code extensions for debugging Python*.
2. View → Command palette → search for *Python: Select Interpreter* → pick the option corresponding to the *raylearn* anaconda environment.
3. Open a Python file, go to Run → Start Debugging → when asked for *Debug Configuration*, pick *Python File*.

You can try one of the examples (e.g. example_one_lens.py or example_rotating_cylinder.py) and go to Run → Start Debugging (Default shortcut: F5).

A few tips on debugging:

- Breakpoints can be set by either:
  - clicking the gutter (next to the line number)
  - or Run → Toggle Breakpoint
  - or default shortcut: F9
- When an error or breakpoint is encountered, code execution is paused and you are dropped into debugging mode.
- From the sidebar, you can pick where in the *call stack* you want to debug (i.e. from which function level the variables are available).
- In the debug console, you can execute one-line Python code to inspect what is going on.
- Debug commands like: *Step over*, *Step into*, *Step out* and *Continue* are available from the *Run* menu.

## Unit testing

Unit tests are run using the pytest package. To run all tests either:

- Let VS Code run pytest for you:
  
  1. Follow the steps in *Getting started* to set up the Python interpreter.
  2. Either:
     - Go to View → Testing, or the activity bar (= sidebar with icons) → Testing (erlenmeyer icon), and click Run Tests (double triangle icon).
     - Or execute *Test: Run All Tests* from the command palette.
  3. Expand >items in the list to view all of its test results.

  Note: If this doesn't work, pytest might not be set up properly with VS Code. Go to settings (gear icon lower left or Ctrl+,) search for *pytest* and make sure 'Enable testing using pytest' is enabled.

- Run pytest directly from the terminal:
  
  1. Navigate your terminal to the directory of the raylearn repository: `cd path/to/raylearn`
  2. Run `conda activate raylearn` if you haven't done so already.
  3. Run `pytest`.
  4. Pytest should execute all tests, and report how many tests passed and failed (and the error message if they failed).

Note: The *interpolate shader* is currently broken on Windows.

## Using a different editor

If you prefer to use a different code editor/IDE, please apply the following settings (if the editor supports them):
Maximum linewidth/textwidth/column: 100
