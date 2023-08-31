% Load single SLM pattern

% Load calibration
calibrationdata = load("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\calibration\calibration_matrix_parabola\calibration_values.mat");
offset_center_slm = calibrationdata.sopt.offset_center_slm;

%% Load pattern
set_RT_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-bottom-λ808.0nm.mat")

%% Random pattern to annihilate the focus
slm.setRect(1, [0 0 1 1]); slm.setData(1, 255*rand(300)); slm.update

%% Flat pattern
slm.setRect(1, [0 0 1 1]); slm.setData(1, 1); slm.update


%% Functions
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
