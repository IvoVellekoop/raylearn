% ===== Directory Configuration ===== %
% Script for setting relative directories.
%
% Future users, please:
%  - Use variables set in this script rather than 'hard coding'
%    directories in scripts, to keep things in one place.
%  - Please keep it platform independent.
%  - Use relative paths, for portability.
%  - Keep stuff within the main directory, for portability.
%  - If this is impossible or impractical, consider using symlinks.
%  - If something should be excluded from git,
%    i.e. generated data, use .gitignore.
%  - If you keep your code on a cloud service such as Dropbox, you can also
%    exclude directories. Which is probably useful if you have large
%    data files.


% Location of current m-file as string array
mfilearray = strsplit(string(mfilename('fullpath')), filesep);


% Directory variables
dirs.main = char(join(mfilearray(1:end-2), filesep));         % Allprojects dir
dirs.repo = char(join(mfilearray(1:end-1), filesep));         % repo directory
dirs.simdata = fullfile(char(dirs.repo), 'SimulationData');   % Simulation Data
dirs.expdata = fullfile(char(dirs.repo), 'ExperimentalData'); % Experimental Data
dirs.localdata = fullfile(char(dirs.repo), 'LocalData');      % Local Data
% expdatadir = '//ad.utwente.nl/TNW/BMPI/Users/Daniel Cox/ExperimentalData';

