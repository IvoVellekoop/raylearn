%% Split Measurement Data
% Some measurement data is packed into one big file. This script walks through the data folder
% and calls split_matfile_SLM_segments on each file.

clear; clc

% Define location paths
addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
dirconfig_raylearn

% Settings
inputpattern = 'F:\ScientificData\raylearn-data\TPM\pencil-beams-raw\*\raylearn_pencil_beam*.mat';
outputgroupfolder = 'F:\ScientificData\raylearn-data\TPM\pencil-beams-split';

% Process
filelist = dir(inputpattern);       % List files to be processed
F = length(filelist);

for f = 1:F                         % Loop over files
    % Construct input and output paths
    matfilepath = fullfile(filelist(f).folder, filelist(f).name);
    [~, measurementfolder] = fileparts(matfilepath);
    [~, samplefolder] = fileparts(filelist(f).folder);
    savedir = fullfile(outputgroupfolder, samplefolder, measurementfolder);
    
    % Split the file
    split_matfile_SLM_segments(matfilepath, savedir, sprintf('File: %i/%i', f, F))
end
