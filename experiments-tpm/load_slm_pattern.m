% Load single SLM pattern

% Load calibration
calibrationdata = load("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\calibration\calibration_matrix_parabola\calibration_values.mat");
offset_center_slm = calibrationdata.sopt.offset_center_slm;

%% Load pattern Bottom
slm_rotation_deg = 2.7;
set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\slm-patterns\pattern-0.5uL-tube-bottom-λ808.0nm.mat")

%% Load pattern Center
slm_rotation_deg = 2.7;
set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-center-λ808.0nm.mat")

%% Load pattern Side
slm_rotation_deg = 2.7;
set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-side-λ808.0nm.mat")

%% Random pattern to annihilate the focus
slm.setRect(1, [0 0 1 1]); slm.setData(1, 255*rand(300)); slm.update

%% Flat pattern
slm.setRect(1, [0 0 1 1]); slm.setData(1, 1); slm.update


%% Functions
function set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, matpath)
    patterndata = load(matpath);                        % Load pattern data
    SLM_pattern_rad = patterndata.phase_SLM;       % Pattern in rad
    SLM_pattern_gv = SLM_pattern_rad * 255 / (2*pi);    % Pattern in gray value
    SLM_pattern_gv_rot = imrotate(SLM_pattern_gv, slm_rotation_deg, "bilinear", "crop"); % Rotate
    slm.setRect(1, [offset_center_slm(1) offset_center_slm(2) 1 1]);
    slm.setData(1, SLM_pattern_gv_rot); slm.update;
end


function set_AO_pattern(slm, offset_center_slm, matpath)
    patterndata = load(matpath);
    slm.setRect(1, [offset_center_slm(1) offset_center_slm(2) 1 1]);
    slm.setData(1, patterndata.slm_pattern_gv_optimal); slm.update
end
