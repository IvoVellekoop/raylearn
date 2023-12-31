% Compute FOV and NA as limited by camera size
% and beam size after OBJ2 as limited by NA of OBJ2.
clear; close all; clc

NA = 0.8;
n = 1;

% All distances in meters!

% Focal distances
f1 = 100e-3;
f2 = 100e-3;
f3 = 200e-3;
f4 = 200e-3;
f5 = 150e-3;
f6a = 200e-3;
f6b = 200e-3;
f7 = 300e-3;
f9 = 150e-3;
f10 = 40e-3;

% Objectives
obj1_tubelength = 200e-3;          % Objective standard tubelength
obj1_magnification = 16;           % Objective magnification

obj2_tubelength = 165e-3;          % Objective standard tubelength
obj2_magnification = 20;          % Objective magnification

fobj1 = obj1_tubelength / obj1_magnification;
fobj2 = obj2_tubelength / obj2_magnification;

% Cameras
% https://www.baslerweb.com/en/products/cameras/area-scan-cameras/ace/aca2000-165umnir/
cam_height = 6e-3;
cam_width = 11.3e-3;

% FOV limit computation
FOV_height_imgcam = cam_height * fobj2 / f9;
FOV_width_imgcam = cam_width * fobj2 / f9;
fprintf('\nFOV as limited by height of image cam = %5.1f\xB5m', FOV_height_imgcam*1e6)
fprintf('\nFOV as limited by width  of image cam = %5.1f\xB5m\n', FOV_width_imgcam*1e6)

% NA limit computation
D_NA_limit_height_by_ftcam = cam_height * f9 / f10;     % Entrance pupil limit caused by finite ft cam sensor size
D_NA_limit_width_by_ftcam = cam_width * f9 / f10;     % Entrance pupil limit caused by finite ft cam sensor size
NA_limit_height_by_ftcam = n * sin(atan(D_NA_limit_height_by_ftcam / (2*fobj2)));
NA_limit_width_by_ftcam = n * sin(atan(D_NA_limit_width_by_ftcam / (2*fobj2)));

fprintf('\nNA as limited by height of fourier cam = %.3f', NA_limit_height_by_ftcam)
fprintf('\nNA as limited by width  of fourier cam = %.3f\n', NA_limit_width_by_ftcam)

% Diameter of full NA beam
D_full_NA = 2 * fobj2 * NA/n;
fprintf('\nDiameter of pupil beam = %.1fmm for NA=%.2f\n', D_full_NA*1e3, NA)

% x = 0:0.02:4; y = sin(atan(x)); plot(x, n*y); ylabel('NA'); xlabel('D / 2f'); grid on


