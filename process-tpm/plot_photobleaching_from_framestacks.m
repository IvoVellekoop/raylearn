%%

framestack_raw_1 = load_framestack('/mnt/bmpi/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/TPM-3D-scans/scans-AO-vs-RT-5mW-try2/tube-500nL-zoom9-no-correction-1_00001.tif');
framestack_raw_2 = load_framestack('/mnt/bmpi/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/TPM-3D-scans/scans-AO-vs-RT-5mW-try2/tube-500nL-zoom9-no-correction-2_00001.tif');
framestack_raw_3 = load_framestack('/mnt/bmpi/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/TPM-3D-scans/scans-AO-vs-RT-5mW-try2/tube-500nL-zoom9-no-correction-3_00001.tif');
framestacks_raw = [framestack_raw_1(:) framestack_raw_2(:) framestack_raw_3(:)];

%%
percentiles = 99.991:0.002:99.999;

high_signals = prctile(framestacks_raw, percentiles)';

%% Plot
plot(high_signals, '.-');
xticks([1 2 3])
xticklabels(["Before", "Halfway", "After"])
ylabel('Signal')
title('Percentiles of flat pattern measurements')
legend(strcat(num2str(percentiles'), repmat("%tile", [numel(percentiles) 1])))
set(gca, 'FontSize', 14)

%%
function framestack_raw = load_framestack(tifpath)
    %% Load frames
    tifinfo = imfinfo(tifpath);         % Get info about image
    num_of_frames = length(tifinfo);    % Number of frames
    framestack = zeros(tifinfo(1).Width, tifinfo(1).Height, num_of_frames); % Initialize
    
    for n = 1:num_of_frames
        framestack(:,:,n) = imread(tifpath, n);
    end
    
    framestack_raw = single(framestack);
end