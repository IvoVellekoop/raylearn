% Compare scanimage frames with and without correction

% Scan settings
hSI.hStackManager.numSlices = 120;
hSI.hStackManager.stackZStepSize = 1.5;
hSI.hChannels.loggingEnable = true;                 % True = Enable save

zoom = hSICtl.hModel.hRoiManager.scanZoomFactor;    % Get zoom
basefilename = sprintf('tube-500nL-zoom%i-', zoom);

% Save directory
hSI.hScan2D.logFilePath = 'C:\LocalData\raylearn-data\TPM\TPM-3D-scans';
fprintf('\nSaving in:\n%s\n\n', hSI.hScan2D.logFilePath)

% No correction
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

% Adaptive optics top pattern
set_AO_pattern(slm, '\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\08-Jun-2023-tube500nL-top\tube_ao_739045.726853_tube500nL-top\tube_ao_739045.726853_tube500nL-top_optimal_pattern.mat')
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-AO');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

% Adaptive optics bottom pattern
set_AO_pattern(slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\08-Jun-2023-tube500nL-bottom\tube_ao_739045.687979_tube500nL\tube_ao_739045.687979_tube500nL_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-AO');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

% No correction
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

% Ray traced top pattern
set_RT_pattern(slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-top-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-RT');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

% Ray traced bottom pattern
set_RT_pattern(slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-bottom-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-RT');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

% No correction
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction-3');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

% Random pattern to annihilate the focus
slm.setRect(1, [0 0 1 1]); slm.setData(1, 255*rand(300)); slm.update


function set_RT_pattern(slm, matpath)
    patterndata = load(matpath);
    SLM_pattern = (angle(patterndata.field_SLM) + pi) * 255 / (2*pi);
    slm.setRect(1, [0 0 1 1]); slm.setData(1, SLM_pattern); slm.update
end


function set_AO_pattern(slm, matpath)
    patterndata = load(matpath);
    slm.setRect(1, [0 0 1 1]); slm.setData(1, patterndata.slm_pattern_gv_optimal); slm.update
end