% Compare scanimage frames with different correction patterns

abort_if_required(hSI, hSICtl)

% Scan settings
hSI.hStackManager.numSlices = 120;
hSI.hStackManager.stackZStepSize = 1.4;
hSI.hChannels.loggingEnable = true;                 % True = Enable save

zoom = hSICtl.hModel.hRoiManager.scanZoomFactor;    % Get zoom
zstep = hSI.hStackManager.stackZStepSize;
basefilename = sprintf('tube-500nL-zoom%i-zstep%.3fum-', zoom, zstep);
savedir = ['C:\LocalData\raylearn-data\TPM\TPM-3D-scans\' sprintf('%s_tube-500nL', date)];

calibrationdata = load("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\calibration\calibration_values.mat");
offset_center_slm = calibrationdata.sopt.offset_center_slm;

% Save directory
mkdir(savedir);
hSI.hScan2D.logFilePath = savedir;
fprintf('\nSaving in:\n%s\n\n', hSI.hScan2D.logFilePath)

%% No correction
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics top pattern
set_AO_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\230620-tube500nL-top\tube_ao_739059.828687_tube500nL-top\tube_ao_739059.828687_tube500nL-top_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-AO-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics bottom pattern
set_AO_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\230620-tube500nL-bottom\tube_ao_739059.875192_tube500nL-bottom\tube_ao_739059.875192_tube500nL-bottom_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-AO-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% No correction
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced top pattern
set_RT_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-top-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-RT-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced bottom pattern
set_RT_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-bottom-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-RT-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% No correction
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction-3');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics top pattern
set_AO_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\230620-tube500nL-top\tube_ao_739059.828687_tube500nL-top\tube_ao_739059.828687_tube500nL-top_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-AO-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics bottom pattern
set_AO_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\230620-tube500nL-bottom\tube_ao_739059.875192_tube500nL-bottom\tube_ao_739059.875192_tube500nL-bottom_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-AO-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% No correction
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction-4');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced top pattern
set_RT_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-top-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-RT-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced bottom pattern
set_RT_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-bottom-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-RT-3');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% No correction
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction-5');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

% Random pattern to annihilate the focus
slm.setRect(1, [0 0 1 1]); slm.setData(1, 255*rand(300)); slm.update


function set_RT_pattern(slm, offset_center_slm, matpath)
    patterndata = load(matpath);
    SLM_pattern = flip((angle(patterndata.field_SLM)) * 255 / (2*pi));
    slm.setRect(1, [offset_center_slm(1) offset_center_slm(2) 1 1]);
    slm.setData(1, SLM_pattern); slm.update;
end


function set_AO_pattern(slm, offset_center_slm, matpath)
    patterndata = load(matpath);
    slm.setRect(1, [offset_center_slm(1) offset_center_slm(2) 1 1]);
    slm.setData(1, patterndata.slm_pattern_gv_optimal); slm.update
end


function abort_if_required(hSI, hSICtl)
    %% Press abort button if required
    if ~strcmp(hSI.acqState, 'idle')
        disp('Scanimage not idle. Trying to abort...')
        hSICtl.abortButton
        pause(0.2)
        if ~strcmp(hSI.acqState, 'idle')
            error('Could not abort current scanimage operation')
        end
        disp('Succesfully aborted current operation.')
    end
end