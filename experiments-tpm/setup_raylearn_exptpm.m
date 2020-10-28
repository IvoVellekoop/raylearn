% Setup paths and hardware for raylearn experiments in Two Photon Microscope

% Define location paths
addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
dirconfig_raylearn

%% Add repo paths
addrepo(fullfile(dirs.main, 'raylearn'))
addrepo(fullfile(dirs.main, 'hardware'))

%% Setup hardware
active_devices.slm = true;
active_devices.galvos = true;
active_devices.cam_img = true;
active_devices.cam_ft = true;
active_devices.pmt_gain = false;
active_devices.sample_stage = false;

setup_hardware
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
