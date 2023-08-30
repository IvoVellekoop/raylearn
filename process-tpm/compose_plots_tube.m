% Compose plot to compare different framestacks
% This script processes the output from plot_report_framestacks
close all; clc;

forcereset = 0;

if forcereset || ~exist('dirs', 'var')
    close all; clear; clc

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn
end

do_save = 1;
savedir = fullfile(dirs.repo, 'plots/3D-scans/agar/');
mkdir(savedir);


%% Compositions
compose_plots(savedir, "_hori-slice", dirs)
compose_plots(savedir, "_max-proj", dirs)

% Settings per subplot
function compose_plots(savedir, suffix, dirs)
    fig = figure;
    set(fig, 'Position', [10, 500, 1200, 800]);
    colormap inferno

    % Global parameters
    w = 0.42;           % Width  subplot (will be forced to square) relative to figure
    h = w;              % Height subplot (will be forced to square) relative to figure
    
    % SLM inset parameters
    patterndata = load(fullfile(dirs.expdata, '/raylearn-data/pattern-0.5uL-tube-bottom-λ808.0nm.mat'));
    inset.NA_mask = patterndata.NA_mask_SLM;
    inset.position = [0.008, 0.008, 0.22, 0.22];  % Position (x,y,w,h) relative to subplot height
    inset.cmap = 'twilight';
    inset.clim = [0 2*pi];
    inset.modvalue = 2*pi;

    ao_dir = '/raylearn-data/TPM/adaptive-optics';


    % No correction
    s1.title = "no correction";
    s1.filename = sprintf("tube-500nL-zoom8-zstep1.400um-no-correction-1_00001%s.fig", suffix);
    s1.pattern_rad = ones(size(inset.NA_mask));
    s1.savedir = savedir;
    s1.position = [1/6-w/2+0.01, 1/2-h/2, w, h];
    ax1 = place_subplot(fig, s1);
    cb = colorbar;
    place_slm_inset(fig, ax1, s1, inset, [-0.06 0])

    % AO top
    s2.title = "AO top correction";
    s2.filename = sprintf("tube-500nL-zoom8-zstep1.400um-top-AO-1_00001%s.fig", suffix);
    s2.pattern_rad = load_AO_pattern(fullfile(dirs.expdata, ao_dir, '21-Aug-2023-tube500nL-top/tube_ao_739119.626092_tube500nL-top/tube_ao_739119.626092_tube500nL-top_optimal_pattern.mat'));
    s2.savedir = savedir;
    s2.position = [1/2-w/2, 3/4-h/2, w, h];
    ax2 = place_subplot(fig, s2);
    place_slm_inset(fig, ax2, s2, inset)

    % RT top
    s3.title = "RT top correction";
    s3.filename = sprintf("tube-500nL-zoom8-zstep1.400um-top-RT-1_00001%s.fig", suffix);
    s3.pattern_rad = load_RT_pattern(fullfile(dirs.expdata, 'raylearn-data/pattern-0.5uL-tube-top-λ808.0nm.mat'));
    s3.savedir = savedir;
    s3.position = [5/6-w/2, 3/4-h/2, w, h];
    ax3 = place_subplot(fig, s3);
    place_slm_inset(fig, ax3, s3, inset)

    % AO bottom
    s4.title = "AO bottom correction";
    s4.filename = sprintf("tube-500nL-zoom8-zstep1.400um-bottom-AO-1_00001%s.fig", suffix);
    s4.pattern_rad = load_AO_pattern(fullfile(dirs.expdata, ao_dir, '21-Aug-2023-tube500nL-bottom/tube_ao_739119.716406_tube500nL-bottom/tube_ao_739119.716406_tube500nL-bottom_optimal_pattern.mat'));
    s4.savedir = savedir;
    s4.position = [1/2-w/2, 1/4-h/2, w, h];
    ax4 = place_subplot(fig, s4);
    place_slm_inset(fig, ax4, s4, inset)

    % RT bottom
    s5.title = "RT bottom correction";
    s5.filename = sprintf("tube-500nL-zoom8-zstep1.400um-bottom-RT-1_00001%s.fig", suffix);
    s5.pattern_rad = load_RT_pattern(fullfile(dirs.expdata, 'raylearn-data/pattern-0.5uL-tube-bottom-λ808.0nm.mat'));
    s5.savedir = savedir;
    s5.position = [5/6-w/2, 1/4-h/2, w, h];
    ax5 = place_subplot(fig, s5);
    place_slm_inset(fig, ax5, s5, inset)

    movegui('center')
end


% Plot subplot
function ax_infig = place_subplot(fig, s)
    filepath = fullfile(s.savedir, s.filename);
    subfig = openfig(filepath);
    ax_separate = subfig.findobj('Type', 'Axes');
    ax_infig = copyobj(ax_separate, fig);       % Copy axes from opened figure to desired figure
    close(subfig)                               % Close opened figure
    drawnow
    
    ax_infig.Title.String = s.title;            % Set title
    set(ax_infig, 'Position', s.position)
    drawnow
    add_scalebar('size', [30, 2], 'text', '30', 'ax', ax_infig)

    % Remove pesky ticks
    set(ax_infig, 'XTick', [])
    set(ax_infig, 'YTick', [])
    xlabel(ax_infig, '')
    ylabel(ax_infig, '')
    drawnow
end


function place_slm_inset(fig, ax_subplot, s, inset, insetshift)
    if nargin < 5
        insetshift = [0, 0];
    end
    pattern = mod(s.pattern_rad - s.pattern_rad(end/2, end/2), inset.modvalue);

    boxpos = plotboxpos(ax_subplot);                        % Get plot box position
%     annotation('rectangle', boxpos, 'edgecolor', 'r', 'linestyle', '-.');
    ARfig = fig.Position(4) ./ fig.Position(3);
    xy = boxpos(1:2) + (inset.position(1:2) + insetshift) .* boxpos(4);    % x, y, relative
    wh = boxpos(4) .* inset.position(3:4) .* [ARfig 1];     % width, height, relative
    ax_inset = axes(fig, 'Position', [xy wh]);              % Apply inset position
    imagesc(ax_inset, pattern, 'AlphaData', inset.NA_mask, 'Tag', 'slm_pattern');
    axis(ax_inset, 'off')
    ax_inset.CLim = inset.clim;                             % Set colormap limits
    colormap(ax_inset, inset.cmap)
    drawnow
end


function pattern_rad = load_RT_pattern(matpath)
    patterndata = load(matpath);
    pattern_rad = flip(patterndata.pathlength_SLM);
end


function pattern_rad = load_AO_pattern(matpath)
    patterndata = load(matpath);
    pattern_rad = patterndata.slm_pattern_2pi_optimal;
end
