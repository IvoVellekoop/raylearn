% Test image shifts from raytracer patterns
%  - Laser 5mW at sample plane, 808nm
%  - Initialize variables slm and daqs before running this script
%    with setup_raylearn_exptpm.m
%  - Go to just above surface of sample

abort_if_required(hSI, hSICtl)

% Scan settings
hSI.hStackManager.numSlices = 80;                   % Number of z-slices
hSI.hStackManager.stackZStepSize = 0.25;            % Step size in um
hSI.hChannels.loggingEnable = true;                 % True = Enable save
hSICtl.hModel.hRoiManager.scanZoomFactor = 12;      % Zoom
pmt_gain_V = 0.60;                                  % PMT gain in Volt
p = struct();
p.backlash_piezo_z_um = -10;                        % Backlash compensation movement piezo

% SLM pattern settings
p.slm_rotation_deg = 0;
calibrationdata = load("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\calibration\calibration_matrix_parabola\calibration_values.mat");
p.offset_center_slm = calibrationdata.sopt.offset_center_slm;

% Retrieve used settings from GUI
p.zoom = hSICtl.hModel.hRoiManager.scanZoomFactor;    % Get zoom
p.zstep = hSI.hStackManager.stackZStepSize;
p.initial_objective_piezo_um = hSI.hMotors.motorPosition;

basefilename = sprintf('beads-zoom%i-zstep%.3fum-', p.zoom, p.zstep);
savedir = ['C:\LocalData\raylearn-data\TPM\TPM-3D-scans\' sprintf('%s_beads_testshift', date)];

% Save directory
mkdir(savedir);
hSI.hScan2D.logFilePath = savedir;
fprintf('\nSaving in:\n%s\n\n', hSI.hScan2D.logFilePath)

%% Flat pattern
hSI.hMotors.motorPosition = p.initial_objective_piezo_um + [0 0 p.backlash_piezo_z_um];
pause(0.5)
hSI.hMotors.motorPosition = p.initial_objective_piezo_um;
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced x-shift
hSI.hMotors.motorPosition = p.initial_objective_piezo_um + [0 0 p.backlash_piezo_z_um];
pause(0.5)
hSI.hMotors.motorPosition = p.initial_objective_piezo_um;
set_RT_pattern(slm, p.slm_rotation_deg, p.offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-focusshift_x30um_y0um_z0um_λ808nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'xshift');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced y-shift
hSI.hMotors.motorPosition = p.initial_objective_piezo_um + [0 0 p.backlash_piezo_z_um];
pause(0.5)
hSI.hMotors.motorPosition = p.initial_objective_piezo_um;
set_RT_pattern(slm, p.slm_rotation_deg, p.offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-focusshift_x0um_y30um_z0um_λ808nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'yshift');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced z-shift + objective z-shift 30um out of sample
hSI.hMotors.motorPosition = p.initial_objective_piezo_um + [0 0 -30] + [0 0 p.backlash_piezo_z_um];
pause(0.5)
hSI.hMotors.motorPosition = p.initial_objective_piezo_um + [0 0 -30];
set_RT_pattern(slm, p.slm_rotation_deg, p.offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-focusshift_x0um_y0um_z30um_λ808nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'zshift_with_obj_zshift');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);
hSI.hMotors.motorPosition = p.initial_objective_piezo_um;


%% Random pattern to annihilate the focus
slm.setRect(1, [0 0 1 1]); slm.setData(1, 255*rand(300)); slm.update


%% Functions
function set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, matpath)
    patterndata = load(matpath);                        % Load pattern data
    SLM_pattern_rad = patterndata.phase_SLM;            % Pattern in rad
    SLM_pattern_gv = SLM_pattern_rad * 255 / (2*pi);    % Pattern in gray value
    SLM_pattern_gv_rot = imrotate(SLM_pattern_gv, slm_rotation_deg, "bilinear", "crop"); % Rotate
    slm.setRect(1, [offset_center_slm(1) offset_center_slm(2) 1 1]);
    slm.setData(1, SLM_pattern_gv_rot); slm.update;
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
