"""
Define directory configuration

Script for setting directories in one central place.
E.g. if your experimental data moves, you can update its path here.

Paths are defined as Path objects from the pathlib built-in library, making platform independent
path manipulation and path related system calls easy.

If you want to modify these paths for local use only:
- Feel free to override the defined paths with manual paths, e.g.:
  Path('C:\\here\\is\\my\\datafolder')
- Use path variables set in this script rather than changing paths manually in
  other scripts/functions, to keep things in one place.
- However, please don't git-push changes that only work on your computer.

If you plan to modify this for global use:
- Use path variables set in this script rather than changing paths manually in other
  scripts/functions, to keep things in one place.
- Please keep it platform independent.
- Use relative paths, for portability.
- When appropriate, keep stuff within the main directory, for portability.
- If this is impossible or impractical, consider using symlinks.
- If something should be excluded from git, i.e. generated data, use .gitignore.
- If you keep your code on a cloud service such as Dropbox, you can also exclude
  directories. Which is probably useful if you have large data files.
"""

from pathlib import Path


dirs = {
    "repo": Path.cwd(),
    "main": Path.cwd().parent,
    "simdata": Path('//ad.utwente.nl/TNW/BMPI/Data/Daniel Cox/SimulationData'),
    "expdata": Path('//ad.utwente.nl/TNW/BMPI/Data/Daniel Cox/ExperimentalData'),
    "localdata": Path.home().joinpath("LocalData"),
}
