% Compare scanimage frames with and without correction

% N = 5;  % Number of times to switch between
% starttime = now;
% for n=1:N
%     n_frames = 1;

% frame_no_correction = grabSIFrame(hSI, hSICtl, 1);
%%
SLM_pattern = -(angle(field_SLM) + pi) * 255 / (2*pi);
%%
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
%% 1.9
frame_no_correction = hSI.hDisplay.lastFrame{1};
disp('Grabbed frame no correction')

%%
slm.setRect(1, [0 0 1 1]); slm.setData(1, SLM_pattern); slm.update

%%
% 3.9
frame_with_correction = hSI.hDisplay.lastFrame{1};
disp('Grabbed frame with correction')

% frame_with_correction = grabSIFrame(hSI, hSICtl, n_frames);
% 
%     eta(n, N, starttime, 'console', 'Grabbing frames...', 0);
% end
%%
starttime = now;
N = 10;
for n=1:N
    try
        frames(:,:,n) = imread('C:\LocalData\no_correction_00007.tif',n);
    end
    eta(n, N, starttime, 'console', 'Loading...', 0);
end

figure; imagesc(max(frames, [], 3))