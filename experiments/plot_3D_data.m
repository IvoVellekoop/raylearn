addpath /home/daniel/NTFS-HDD/Code-big/utilities/

%% Load data
load('/home/daniel/Dropbox/Private/Code/Python/raylearn/sim.mat')
load /home/daniel/NTFS-HDD/ScientificData/raylearn-data/raylearn_led_1x400um_21-Aug-2020_738024.796382.mat; sourcename='LED';

%% Sim
figure
im3((sim),...
    'slicedim', 2,...
    'maxprojection', 1,...
    'dimlabels', {'y (µm)','x (µm)','z (µm)'},...
    'dimdata', {ys_um, xs_um, zs_um},...
    'title', 'Simulation')
fig_resize(400,2)
saveas(gcf, 'plots/sim_side.png')

figure
im3((sim),...
    'slicedim', 3,...
    'slicenum', 125,...
    'dimlabels', {'y (µm)','x (µm)','z (µm)'},...
    'dimdata', {ys_um, xs_um, zs_um},...
    'title', 'Simulation')
fig_resize(400,2);
saveas(gcf, 'plots/sim_xy.png')

%% Measurement

% Parameters
framenum = 455;             %%% Manually choose this for now
xysize = 60;                % Match raylearn setting
pixsize = 4.8;              % Camera pixel size (µm)

% Compute magnification from focal distances
[Ny, Nx, Nz] = size(scan3D);
fobj = 1.65;                % Objective effective focal distance (mm)
f3 = 75;
f4 = 150;
f5 = 50;
Mtr  = f3*f5 / (fobj*f4);   % Transverse Magnification
Mlat = Mtr^2;               % Lateral magnification

Mtr_pixsize = pixsize / Mtr;
xymax_um = 0.5 * Ny * Mtr_pixsize;            % Maximum x or y from center

xyrange = (Mtr_pixsize-xymax_um):Mtr_pixsize:xymax_um;


% Select interesting part
zrange_scan3D = p.scanrange_um / Mlat;
zselect = Nz:-1:floor(Nz/2);
yselect = 100:260;
xselect = 180:340;

figure
im3(flip(scan3D(yselect, xselect, zselect), 3),...
    'slicedim', 2,...
    'maxprojection', 1,...
    'dimlabels', {'y (µm)','x (µm)','z (µm)'},...
    'dimdata', {xyrange(yselect), xyrange(xselect), zrange_scan3D(zselect)},...
    'title', ['Measurement ' sourcename])
saveas(gcf, 'plots/LED_side.png')


% Select interesting part
yselect = 120:240;
xselect = 190:310;

figure
im3(flip(scan3D(yselect, xselect, zselect), 3),...
    'slicedim', 3,...
    'slicenum', 80,...
    'dimlabels', {'y (µm)','x (µm)','z (µm)'},...
    'dimdata', {xyrange(yselect), xyrange(xselect), zrange_scan3D(zselect)},...
    'title', ['Measurement ' sourcename])
saveas(gcf, 'plots/LED_xy.png')
