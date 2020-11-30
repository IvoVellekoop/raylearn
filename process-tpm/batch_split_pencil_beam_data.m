% Split Measurement Data
% Some measurement data is packed into one big file. This script walks through the data folder
% and calls split_matfile_SLM_segments on each file.

clear; clc

% Define location paths
addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
dirconfig_raylearn

% Settings
inputpattern = 'F:\ScientificData\raylearn-data\TPM\pencil-beams-raw\*\raylearn_pencil_beam*.mat';
outputgroupfolder = 'F:\ScientificData\raylearn-data\TPM\pencil-beams-split';
consoletext = 'Splitting data files...';

% Process
filelist = dir(inputpattern);       % List files to be processed
F = length(filelist);

% Compute total size of files
bytestotal = 0;
for f = 1:F
    bytestotal = bytestotal + filelist(f).bytes;
end

% Loop over files and split
disp(consoletext)
starttime = now;
bytesdone = 0;
for f = 1:F
    % Construct input and output paths
    matfilepath = fullfile(filelist(f).folder, filelist(f).name);
    [~, measurementfolder] = fileparts(matfilepath);
    [~, samplefolder] = fileparts(filelist(f).folder);
    savedir = fullfile(outputgroupfolder, samplefolder, measurementfolder);
    
    % Count size of processed files
    bytesdone = bytesdone + filelist(f).bytes;
    
    % Split the file
    split_matfile_SLM_segments(matfilepath, savedir)
    
    eta(bytesdone, bytestotal, starttime, 'cmd', sprintf('%s\nFiles done: %i/%i', consoletext, f, F), 0);
end
