%=== Raylearn Measurements ===
% Measurements:
% - Laser: FT plane field, FT plane intensity, IMG plane 3D scan focus
% - LED:   FT plane intensity, IMG plane 3D scan focus

do_reset = 1;
if do_reset
    close all; clear; clc;  % Clear memory
    setup_raylearn_exp      % Setup hardware
end

%% Settings
% 3D scan settings
p.samplename = '1x400um';
p.long_exposure_time_us = 100e3;            % longest exposure in us
savedir = fullfile(expdatadir);
p.scanrange_um = (25e3:5e1:50e3);           % scan range in µm
p.preview_pos_um = 40e3;                    % preview scan range in µm
img_min_exposure_time_us = 59;              % Minimum exposure time in µs for img cam

%% Check some settings
fprintf('\nSample name set to <strong>%s</strong>. Is this correct?\nPress Enter to continue...\n', p.samplename)

%% === Background images ===
fprintf('\n<strong>Background image</strong>\n')
disp(' - Turn off all light sources')
disp('Press Enter to continue...')
pause

% Capture and show background images
cam_ft.trigger()
bg_ft = single(cam_ft.getData());
cam_img.setExposureTime(p.long_exposure_time_us);
cam_img.trigger()
bg_img = single(cam_img.getData());

% Show backround image
fig = figure; set(fig,'KeyPressFcn', @keyfunc_unpause);
subplot(1,2,1); imagesc(bg_ft);  colorbar; title('Fourier Cam background')
subplot(1,2,2); imagesc(bg_img); colorbar; title(sprintf('Image Cam background\nPress Enter to continue...'))
set(fig, 'Position', [350 150 800 300])

disp('Captured backround images')
disp('Press Enter to continue...')
waitfor(fig); clc

%% Laser - Fourier plane field
fprintf('\n<strong>Fourier plane field measurement - %s</strong>\n', p.samplename)
disp(' - Block the image plane camera.')
disp(' - Unblock reference beam.')
disp(' - Check objective alignment.')
disp(' - Adjust laser intensity for interferometry.')
disp('Press Enter on figure to continue...')

% Flat phase pattern for checking objective alignment
patch_id = 1;
slm.setData(patch_id, 0); slm.update;

% Show live camera to check objective alignment
fig = live_cam(cam_ft, 'Check Objective alignment', 0, 1);

% Blocked phase pattern SLM for adjusting intensity
slm.setRect(patch_id, [0 0 1 1]);
slm.setData(patch_id, 255*rand(42)); slm.update;

% Show live camera to adjust laser intensity
fig = live_cam(cam_ft, 'Adjust laser intensity', 0.5, 0.8);
slm.setData(patch_id, 0); slm.update;

% Take phase stepping measurement
fprintf('\nStarting phase stepping measurement...\n')
[field_ft, phase_step_frames_ft] = phase_step_measurement(cam_ft, camopt_ft, slm, 4);

fig = figure; set(fig,'KeyPressFcn', @keyfunc_unpause);
imagesc(complex2rgb(field_ft, 'vgamma', 0.5))
title('Field measurement - Press Enter to continue...'); axis image;
disp('Field Measurement done.')
disp('Press Enter to continue...')
waitfor(fig); clc

%% Laser - Fourier plane intensity
fprintf('\n<strong>Fourier plane intensity measurement - Laser</strong>\n')
disp(' - Block reference beam')
disp(' - Block image plane camera')
disp('Press Enter to continue...')

% Show live camera to adjust laser intensity
fig = live_cam(cam_ft, 'Block reference beam', 0.5, 0.8);

cam_ft.trigger();
intensity_ft = single(cam_ft.getData());
fig = figure; set(fig,'KeyPressFcn', @keyfunc_unpause);
imagesc(intensity_ft)
title('Intensity measurement - Press Enter to continue...'); axis image;
waitfor(fig); clc


 %% Laser - 3D scan
fprintf('\n<strong>3D scan of focus - Laser - %s</strong>\n', p.samplename)
disp(' - Check whether the laser light is sufficiently')
disp('   diminished with ND filters. (NE30A or NE40A  for img cam).')
fprintf('\nPress Enter to continue...\n\n')
pause

% Perform preview scan
try mot_cam.moveTo(p.preview_pos_um); catch err; warning(err.message); end
cam_img.setExposureTime(p.long_exposure_time_us);
fig = live_cam(cam_img, 'Adjust laser intensity', 0.40, 0.60);
clc;

% Perform scan
[scan3D, thresholds] = motor_cam_scan(mot_cam, p.scanrange_um, cam_img, p.long_exposure_time_us, bg_img, img_min_exposure_time_us);

%% Laser - Show the result
fig = show_scan(scan3D, p, ['Laser focus ' p.samplename]);
waitfor(fig); clc

%% Laser - Save measurement   %%%% move to subfunction?
savepath = fullfile(savedir, sprintf('raylearn_laser_%s_%s_%f.mat', p.samplename, date, now));
fprintf('\nSaving to %s\n', savepath)
save(savepath, '-v7.3', 'scan3D', 'thresholds', 'p', 'bg_ft', 'bg_img',...
    'field_ft', 'phase_step_frames_ft', 'intensity_ft')


%% LED - Fourier plane intensity
fprintf('\n<strong>Fourier plane intensity measurement - LED</strong>\n')
disp(' - Block the laser.')
disp(' - Turn on the LED.')
disp(' - Check the flip mirrors for using the LED.')
disp(' - Block image plane camera')
disp('Press Enter to continue...')
pause

cam_ft.trigger();
intensity_ft = single(cam_ft.getData());
fig = figure; set(fig,'KeyPressFcn', @keyfunc_unpause);
imagesc(intensity_ft)
title('Intensity measurement'); axis image;
waitfor(fig); clc

%% LED - 3D scan instructions
fprintf('\n<strong>3D scan of focus - LED - %s</strong>\n', p.samplename)
disp(' - Remove ND filter of image plane cam.')
fprintf('\nPress Enter to continue...\n\n')
pause

% Perform preview scan
try mot_cam.moveTo(p.preview_pos_um); catch err; warning(err.message); end
cam_img.setExposureTime(p.long_exposure_time_us);
fig = live_cam(cam_img, 'To start scan,', 0, 1);
clc;

% Perform scan
[scan3D, thresholds] = motor_cam_scan(mot_cam, p.scanrange_um, cam_img, p.long_exposure_time_us, bg_img, img_min_exposure_time_us);

%% LED - Show the result
titletext = sprintf('3D scan of led focus - %s - max field projection', p.samplename);
fig = show_scan(scan3D, p, titletext);
waitfor(fig); clc

%% LED - Save measurement
savepath = fullfile(savedir, sprintf('raylearn_led_%s_%s_%f.mat', p.samplename, date, now));
fprintf('\nSaving to %s\n', savepath)
save(savepath, '-v7.3', 'scan3D', 'thresholds', 'p', 'bg_ft', 'bg_img', 'intensity_ft')


%% Functions

function fig = live_cam(cam, titlestr, lowest_allowed_max, highest_allowed_max)
    % Show a live camera image and continue on keypress
    fig = figure;
    set(fig,'KeyPressFcn', @keyfunc_stop_preview);  % Keypress function
    global continue_preview;
    continue_preview = true;
    framerate = 0;
    
    while continue_preview      % Keep showing camera until key is pressed
        lastframetime = now;
        cam.trigger();
        oneframe = single(cam.getData());
        
        % Check saturation
        abovehighfraction = mean(oneframe > 65520*highest_allowed_max, [1 2]);
        abovelowfraction  = mean(oneframe > 65520*lowest_allowed_max,  [1 2]);
        if abovehighfraction > 0
            bottomstr = sprintf('Some pixels >%i%% of max; please lower intensity...', highest_allowed_max*100);
        elseif abovelowfraction == 0
            bottomstr = sprintf('No pixels >%i%% of max; please increase intensity...', lowest_allowed_max*100);
        else
            bottomstr = 'Press Enter to continue...';
        end
        
        showcameraimage(oneframe, sprintf(...
            '%s - %.1ffps\n%s', titlestr, framerate, bottomstr))
        
        % Compute framerate
        framerate = 1 / ((now - lastframetime) * 86400);
    end

end


function showcameraimage(oneframe, titlestring)
    % Show camera image with title and stuff
    imagesc(oneframe, [0 65520])
    title(titlestring)
    grid on
    axis image
    drawnow
end


function fig = show_scan(scan, p, titletext)
    % Show 3D scan
    % Show a maximum field projection of the scan.
    % Inputs:
    % scan:      3D num array of with the scan data.
    % titletext: Char array or string with plot title text.
    
    fig = figure; set(fig,'KeyPressFcn', @keyfunc_unpause);
    subplot(2,3,1)
    imagesc(squeeze(max(scan, [], 2)) .^ (1/2), 'xdata', p.scanrange_um);
    xlabel('z (µm)'); colorbar;
    title(sprintf('3D scan - %s - max field projection', titletext))
    
    subplot(2,3,2); imagesc(scan(:,:,round(1/6*end))); colorbar
    subplot(2,3,3); imagesc(scan(:,:,round(2/6*end))); colorbar
    subplot(2,3,4); imagesc(scan(:,:,round(3/6*end))); colorbar
    subplot(2,3,5); imagesc(scan(:,:,round(4/6*end))); colorbar
    subplot(2,3,6); imagesc(scan(:,:,round(5/6*end))); colorbar
    set(fig, 'Position', [50 200 1100 500])
    
    disp('If scan is ok, press Enter to continue...')
end


function keyfunc_stop_preview(src, event)
   % Keypress function to stop camera preview when enter is pressed
   if strcmp(event.Key, 'return')
       global continue_preview;
       continue_preview = false;
       close(src);
   end
end


function keyfunc_unpause(src, event)
   % Keypress function to unpause when enter is pressed
   if strcmp(event.Key, 'return')
       close(src);
   end
end
