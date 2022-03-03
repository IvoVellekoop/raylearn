% ===== Directory Configuration ===== %
% Script for setting relative directories in one central place.
% E.g. if your experimental data moves, you can update its path here.
%
% If you want to modify these paths for local use only:
% - Feel free to override the defined paths with manual paths (e.g. C:\here\is\my\datafolder).
% - Use path variables set in this script rather than putting
%   addpath commands in scripts, to keep things in one place.
% - However, please don't git-push changes that only work on your computer.
%
% If you plan to modify this for global use:
%  - Use variables set in this script rather than putting
%    addpath commands in scripts, to keep things in one place.
%  - Please keep it platform independent.
%  - Use relative paths, for portability.
%  - When appropriate, keep stuff within the main directory, for portability.
%  - If this is impossible or impractical, consider using symlinks.
%  - If something should be excluded from git, i.e. generated data, use .gitignore.
%  - If you keep your code on a cloud service such as Dropbox, you can also exclude
%    directories. Which is probably useful if you have large data files.


% Directory variables
dirs.repo = fileparts(mfilename('fullpath'));                 % repo directory
dirs.main = fileparts(dirs.repo);                             % Allprojects dir
dirs.simdata = '\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\SimulationData';  % Simulation Data
dirs.expdata = '\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData';% Experimental Data
dirs.localdata = 'C:\LocalData';      % Local Data


%% Add repo paths
restoredefaultpath
addrepo(fullfile(dirs.main, 'raylearn'))
addrepo(fullfile(dirs.main, 'utilities'))
addrepo(fullfile(dirs.main, 'hardware'))

fprintf('\n')

%% Subfunctions

function addrepo(basepath)
    % addrepo
    % Add given directory to path with subdirectories, but exclude any .git
    % (sub)directories, after checking whether it exists.
    assert(isfolder(basepath), sprintf('The repository %s could not be found.', basepath))
    path = genpath(basepath);                       % Generate path as normal
    filteredpath = regexprep(path, '[^;]*\.git[^;]*;', ''); % Exclude */.git/*
    addpath(filteredpath);                          % Add to path
    fprintf('Found repo %s.\n', basepath)
end
