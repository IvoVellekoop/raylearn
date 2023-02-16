close all; clear; clc
addpath(fileparts(fileparts(mfilename('fullpath'))))   % Grand parent dir
dirconfig_raylearn
addpath(fullfile(dirs.main, 'scanimage'))

lambda = 715;
sopt.displayPort = 1;
    
% setup SLM
[slm, sopt] = SLMsetup(lambda, sopt);
