addpath /home/daniel/NTFS-HDD/Code-big/utilities/

%% Load data
% load('/home/daniel/Dropbox/Private/Code/Python/raylearn/sim.mat')
load /home/daniel/NTFS-HDD/ScientificData/raylearn-data/raylearn_led_1x400um_21-Aug-2020_738024.796382.mat

%% Sim
figure
im3((sim), 'slicedim',2, 'maxprojection',1, 'dimlabels',{'y (µm)','x (µm)','z (µm)'}, 'dimdata', {ys_um, xs_um, zs_um})

%% Measurement

% Parameters
framenum = 455;             %%% Manually choose this for now
xysize = 60;                % Match raylearn setting
pixsize = 4.8;              % Camera pixel size (µm)

% Compute magnification from focal distances
fobj = 1.65;                % Objective effective focal distance (mm)
f3 = 75;
f4 = 150;
f5 = 50;
Mtr  = f3*f5 / (fobj*f4);   % Transverse Magnification
Mlat = Mtr^2;               % Lateral magnification

xymax_um = 0.5 * xysize * pixsize / Mtr;            % Maximum x or y from center

xyrange = [-xymax_um xymax_um];

%
figure
zrange_scan3D = p.scanrange_um / Mlat;
im3((flip(scan3D(:,:,:), 3)), 'slicedim',2, 'maxprojection',1, 'dimlabels',{'y (µm)','x (µm)','z (µm)'}, 'dimdata', {xyrange, xyrange, zrange_scan3D})
