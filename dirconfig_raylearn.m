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
addrepo(fullfile(dirs.main, 'raylearn'), ["__pycache__", "LocalData"]);
addrepo(fullfile(dirs.main, 'utilities'));
addrepo(fullfile(dirs.main, 'hardware/matlab'));

fprintf('\n')

%% Subfunctions

function path_string = addrepo(basepath, skip_folders, folder_depth, path_string)
    % <strong>addrepo</strong>
    % Recursively add path with subdirectories, but exclude any .git subdirectories, method
    % folders (starting with @) and private folders.
    %
    % Quick usage:
    % addrepo('utilities');
    % addrepo('utilities', ["vol3d", "Example_codes_for_functions"]);
    %
    % Input:
    % basepath      Char array or string representing path to directory.
    % skip_folders  String array with subfolders (or subsubfolders, etc) to skip.
    %               Default is empty string.
    % folder_depth  DON'T USE. Used for recursing.
    % path_string   DON'T USE. Used for recursing.
    %
    % Output:
    % path_string   The path string that's added to the Matlab path.
    %
    % % Example code - Note: the first line doesn't work with "Evaluate selection (F9)"
    % % Put this code in a file in the main directory of your repository, or edit accordingly
    % mfilearray = strsplit(string(mfilename('fullpath')), filesep);    % Current directory
    % maindir = char(join(mfilearray(1:end-2), filesep));               % Allprojects dir
    % addrepo(fullfile(maindir, 'utilities'));

    if nargin < 2
        skip_folders = "";
    end

    if nargin < 3
        folder_depth = 0;       % If unspecified, initialize at 0 -> basepath
        path_string = "";       % If unspecified, initialize the path_string
    end

    % Check if it exists
    assert(isfolder(basepath), sprintf('The repository %s could not be found.', basepath))
    path_string = strcat(path_string, ":", basepath);   % Build the path_string

    dirstruct = dir(basepath);                          % List subfolders and files
    D = length(dirstruct);                              % Number of files and subfolders

    % Recurse through folder structure and add subdirectories, except for .git folder
    for d = 1:D
        itemname = dirstruct(d).name;
        itempath = fullfile(dirstruct(d).folder, itemname);

        % Check if item is a folder that should be recursed into
        if ~any(itemname == [".", "..", ".git", "private"]) ... % Skip current/parent/.git/private
                    && ~any(itemname == skip_folders) ...       % Skip items in skip list
                    && itemname(1) ~= "@" ...                   % Skip method folders
                    && itemname(1) ~= "+" ...                   % Skip package folders
                    && dirstruct(d).isdir                       % Skip non-folders

            % Recurse into subfolder
            path_string = addrepo(itempath, skip_folders, folder_depth+1, path_string);
        end
    end

    % If recursion is at basepath, we're done. Add the path_string to the Matlab path.
    if folder_depth == 0
        addpath(strcat(path_string, ":"))
        fprintf('Found repo %s.\n', basepath)
    end
end
