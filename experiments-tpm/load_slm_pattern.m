% Load single SLM pattern

p.slm_rotation_deg = 3.4;                   % Can be found with an SLM coordinate calibration
p.scale_slm_x = 0.9827;                     % Scale SLM 'canvas' by this amount in x
p.scale_slm_y = 0.9557;                     % Scale SLM 'canvas' by this amount in y

% Load calibration
calibrationdata = load("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\calibration\calibration_matrix_parabola\calibration_values.mat");
p.offset_center_slm = calibrationdata.sopt.offset_center_slm;

p.system_aberration_data = load("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\28-Sep-2023-system-aberration\tube_ao_739157.585029_system-aberration\tube_ao_739157.585029_system-aberration_optimal_pattern.mat");
p.system_aberration_pattern = p.system_aberration_data.slm_pattern_gv_optimal;

%% Load pattern Bottom
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\slm-patterns\pattern-0.5uL-tube-bottom-λ808.0nm.mat")

%% Load pattern Center
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\slm-patterns\pattern-0.5uL-tube-center-λ808.0nm.mat")

%% Load pattern Side
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\slm-patterns\pattern-0.5uL-tube-side-λ808.0nm.mat")

%% Load pattern Side with tilt
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\slm-patterns\pattern-0.5uL-tube-side-with-tilt-λ808.0nm.mat")

%% Random pattern to annihilate the focus
slm.setRect(1, [0 0 1 1]); slm.setData(1, 255*rand(300)); slm.update

%% Flat pattern
slm.setRect(1, [0 0 1 1]); slm.setData(1, 1); slm.update


%% Functions
function set_RT_pattern(slm, p, matpath)
    patterndata = load(matpath);                        % Load pattern data
    SLM_pattern_rad = -patterndata.phase_SLM';            % Pattern in rad
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
    SLM_pattern_gv_rot_scaled_bg = SLM_pattern_gv_rot_scaled + p.system_aberration_pattern;

    slm.setRect(1, [p.offset_center_slm(1) p.offset_center_slm(2) 1 1]);
    slm.setData(1, SLM_pattern_gv_rot_scaled_bg); slm.update;
end


function set_AO_pattern(slm, p, matpath)
    patterndata = load(matpath);
    slm.setRect(1, [p.offset_center_slm(1) p.offset_center_slm(2) 1 1]);
    slm.setData(1, patterndata.slm_pattern_gv_optimal); slm.update
end
