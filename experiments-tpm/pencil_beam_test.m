
bg_patch_id = 2;                    % Background grating Patch ID
segment_patch_id = 3;               % Pencil Beam segment SLM patch ID

diameter = 0.30;                    % diameter of circular SLM geometry
slm_offset_x = 0.00;                % horizontal offset of rectangle SLM geometry
slm_offset_y = 0.00;                % vertical offset of rectangle SLM geometry
N_diameter = 8;                     % number of segments on SLM diameter

% Set background grating
[NxSLM, NySLM] = size(slm.getPixels);
slm.setRect(bg_patch_id, [0 0 1 1]);
slm.setData(bg_patch_id, bg_grating('blaze', 4, 0, 255, NySLM));

% Set pencil beam segment
[rects, N] = BlockedCircleSegments(N_diameter, diameter, slm_offset_x, slm_offset_y, 0, 1);

for n = 1:N
    slm.setRect(segment_patch_id, rects(n,:));
    slm.setData(segment_patch_id, 0);
    slm.update
    pause(0.5)
end


slm.setData(bg_patch_id, bg_grating('blaze', -100, 0, 255, NySLM)); slm.update

%% Move galvos
X = 0;
Y = 0;
outputSingleScan(daqs, [X,Y]);


