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
savedir = fullfile(dirs.repo, 'plots/3D-scans/');
mkdir(savedir);

%%
compose_plots(savedir, "_hori-slice")
compose_plots(savedir, "_max-proj")

% Settings per subplot
function compose_plots(savedir, suffix)
    fig = figure;
    set(fig, 'Position', [10, 500, 1200, 800]);
    colormap inferno

    % Global parameters
    w = 0.42;
    h = 0.42;

    % No correction
    s1.title = "no correction";
    s1.filename = sprintf("tube-500nL-zoom8-zstep1.400um-no-correction-1_00001%s.fig", suffix);
    s1.savedir = savedir;
    s1.position = [1/6-w/2+0.01, 1/2-h/2, w, h];
    place_subplot(fig, s1)
    cb = colorbar;

    % AO top
    s2.title = "AO top correction";
    s2.filename = sprintf("tube-500nL-zoom8-zstep1.400um-top-AO-1_00001%s.fig", suffix);
    s2.savedir = savedir;
    s2.position = [1/2-w/2, 3/4-h/2, w, h];
    place_subplot(fig, s2)

    % RT top
    s3.title = "RT top correction";
    s3.filename = sprintf("tube-500nL-zoom8-zstep1.400um-top-RT-1_00001%s.fig", suffix);
    s3.savedir = savedir;
    s3.position = [5/6-w/2, 3/4-h/2, w, h];
    place_subplot(fig, s3)

    % AO bottom
    s4.title = "AO bottom correction";
    s4.filename = sprintf("tube-500nL-zoom8-zstep1.400um-bottom-AO-1_00001%s.fig", suffix);
    s4.savedir = savedir;
    s4.position = [1/2-w/2, 1/4-h/2, w, h];
    place_subplot(fig, s4)

    % RT bottom
    s5.title = "RT bottom correction";
    s5.filename = sprintf("tube-500nL-zoom8-zstep1.400um-bottom-RT-1_00001%s.fig", suffix);
    s5.savedir = savedir;
    s5.position = [5/6-w/2, 1/4-h/2, w, h];
    place_subplot(fig, s5)

    movegui('center')
end

% Plot subplot
function place_subplot(fig, s)
    filepath = fullfile(s.savedir, s.filename);
    subfig = openfig(filepath);
    ax_separate = gca;
    subplot(ax_separate)                        % Turn the axes in to a subplot axes
    ax_infig = copyobj(ax_separate, fig);       % Copy axes from opened figure to desired figure
    close(subfig)                               % Close opened figure
    drawnow
    
    ax_infig.Title.String = s.title;            % Set title
    set(ax_infig, 'Position', s.position)
    add_scalebar('size', [30, 2], 'text', '30', 'ax', ax_infig)

    set(ax_infig, 'XTick', [])
    set(ax_infig, 'YTick', [])
    xlabel(ax_infig, '')
    ylabel(ax_infig, '')
end


