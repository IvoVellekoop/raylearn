% Test image shifts from raytracer patterns
%  - Laser 5mW at sample plane, 808nm
%  - Initialize variables slm and daqs before running this script
%    with setup_raylearn_exptpm.m
%  - Go to just above surface of sample

abort_if_required(hSI, hSICtl)

% Scan settings
hSI.hStackManager.numSlices = 60;                   % Number of z-slices
hSI.hStackManager.stackZStepSize = 0.333;           % Step size in um
hSI.hChannels.loggingEnable = true;                 % True = Enable save
hSICtl.hModel.hRoiManager.scanZoomFactor = 10;      % Zoom
pmt_gain_V = 0.60;                                  % PMT gain in Volt
p = struct();
p.backlash_piezo_z_um = -10;                        % Backlash compensation movement piezo

% SLM pattern settings
p.slm_rotation_deg = 0;
p.scale_slm_x = 0.9852;
p.scale_slm_y = 0.9737;
calibrationdata = load("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\calibration\calibration_matrix_parabola\calibration_values.mat");
p.offset_center_slm = calibrationdata.sopt.offset_center_slm;

% Retrieve used settings from GUI
p.zoom = hSICtl.hModel.hRoiManager.scanZoomFactor;    % Get zoom
p.zstep = hSI.hStackManager.stackZStepSize;
p.initial_objective_piezo_um = hSI.hMotors.motorPosition;

basefilename = sprintf('beads-zoom%i-zstep%.3fum-', p.zoom, p.zstep);
savedir = ['C:\LocalData\raylearn-data\TPM\TPM-3D-scans\' sprintf('%s_beads_testshift_xscale%f_yscale%f', date, p.scale_slm_x, p.scale_slm_y)];

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
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\slm-patterns\pattern-focusshift_x30um_y0um_z0um_λ808nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'xshift');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced y-shift
hSI.hMotors.motorPosition = p.initial_objective_piezo_um + [0 0 p.backlash_piezo_z_um];
pause(0.5)
hSI.hMotors.motorPosition = p.initial_objective_piezo_um;
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\slm-patterns\pattern-focusshift_x0um_y30um_z0um_λ808nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'yshift');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced z-shift + objective z-shift 30um out of sample
hSI.hMotors.motorPosition = p.initial_objective_piezo_um + [0 0 -30] + [0 0 p.backlash_piezo_z_um];
pause(0.5)
hSI.hMotors.motorPosition = p.initial_objective_piezo_um + [0 0 -30];
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\slm-patterns\pattern-focusshift_x0um_y0um_z30um_λ808nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'zshift_with_obj_zshift');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Bring piezo stage back
hSI.hMotors.motorPosition = p.initial_objective_piezo_um + [0 0 p.backlash_piezo_z_um];
pause(0.5)
hSI.hMotors.motorPosition = p.initial_objective_piezo_um;


%% Random pattern to annihilate the focus
slm.setRect(1, [0 0 1 1]); slm.setData(1, 255*rand(300)); slm.update


%% Functions
function set_RT_pattern(slm, p, matpath)
    patterndata = load(matpath);                        % Load pattern data
    SLM_pattern_rad = patterndata.phase_SLM;            % Pattern in rad
    SLM_pattern_gv = SLM_pattern_rad * 255 / (2*pi);    % Pattern in gray value
    SLM_pattern_gv_rot = imrotate(SLM_pattern_gv, p.slm_rotation_deg, "bilinear", "crop"); % Rotate

    % Prepare scaled SLM pattern (slm pattern must be square!)
    Nslmpattern = size(SLM_pattern_gv_rot, 2);
    x_slmpattern = linspace(-1, 1, Nslmpattern);
    y_slmpattern = x_slmpattern';
    xq = x_slmpattern * p.scale_slm_x;
    yq = y_slmpattern * p.scale_slm_y;
    
    extrapval = 0;
    SLM_pattern_gv_rot_scaled = interp2(x_slmpattern, y_slmpattern, SLM_pattern_gv_rot, xq, yq, 'bilinear', extrapval);

    slm.setRect(1, [p.offset_center_slm(1) p.offset_center_slm(2) 1 1]);
    slm.setData(1, SLM_pattern_gv_rot_scaled); slm.update;
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
