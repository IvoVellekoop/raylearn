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

saveprops.do_save = 1;
saveprops.dir = fullfile(dirs.repo, 'plots/3D-scans/01-Oct-2023_tube-500nL');      % Folder for reading and writing plots
saveprops.save_prefix = 'tube-0.5uL-tube-agar';
mkdir(saveprops.dir);


%% Compositions
compose_plots(saveprops, "_99.5prctile-proj", dirs)

% Settings per subplot
function compose_plots(saveprops, suffix, dirs)
    fig = figure;
    set(fig, 'Position', [10, 500, 1600, 700]);
    colormap inferno
    drawnow

    % Global parameters
    w = 0.42;           % Width  subplot (will be forced to square) relative to figure
    h = w;              % Height subplot (will be forced to square) relative to figure

    global_props = struct();
    global_props.title.FontSize = 13;

    % Focus marker properties
    global_props.focus_marker.MarkerSize = 15;
    global_props.focus_marker.LineWidth = 2;
    global_props.focus_marker.Color = [0.5 1 1];
    
    % SLM inset parameters
    patterndata = load(fullfile(dirs.expdata, '/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-bottom-λ808.0nm.mat'));
    inset.NA_mask = patterndata.NA_mask_SLM;
    inset.position = [0.008, 0.008, 0.22, 0.22];  % Position (x,y,w,h) relative to subplot height
    inset.cmap = 'twilight';
    inset.clim = [0 2*pi];
    inset.modvalue = 2*pi;

    ao_dir = '/raylearn-data/TPM/adaptive-optics';

    % Brightfield
    titlestr = "a. Full tube in bright field";
    ax_brightfield = axes();
    img_brightfield = imread(fullfile(dirs.expdata, "/raylearn-data/Zeiss-brightfield/tube-500nL/zeiss-10x-tube-500nL-5b-cropped-scalebar.png"));
    imagesc(ax_brightfield, img_brightfield)
    title(ax_brightfield, titlestr, 'FontSize', global_props.title.FontSize)
    ax_brightfield.Position = [0.1-w/2, 0.75-h/2, w, h];
    axis image
    set(ax_brightfield, 'XTick', [])
    set(ax_brightfield, 'YTick', [])
    pixel_size_um = 0.62799;
    w_rectangle = 148.07/pixel_size_um;
    h_rectangle = 175/pixel_size_um;
    x_rectangle = size(img_brightfield, 2)/2 - w_rectangle/2;
    y_rectangle = size(img_brightfield, 1)/2 - h_rectangle/2;
    rectangle('Position', [397 370 w_rectangle h_rectangle], 'EdgeColor', [1 0 0])
    drawnow


    % No correction
    s11.titlestr = "f. no correction";
    s11.filename = sprintf("tube-500nL-zoom8-zstep0.500um-no-correction-1_00001%s.fig", suffix);
    s11.pattern_rad = ones(size(inset.NA_mask));
    s11.savedir = saveprops.dir;
    s11.position = [0.1-w/2, 0.25-h/2, w, h];
    ax11 = place_subplot(fig, s11, global_props);
    cb = colorbar(ax11);
    ylabel(cb, 'log_{10}(PMT signal)')
    place_slm_inset(fig, ax11, s11, inset, [0 0])

    % Create colorbar for phase
    drawnow
    pause(1)
    boxpos = plotboxpos(ax11);
    ax_cb_phase = axes();
    imagesc(ax_cb_phase, linspace(0, 2*pi, 256)');

    ARfig = fig.Position(4) ./ fig.Position(3);
    x_cb_phase = boxpos(1) + 0.045;
    h_cb_phase = boxpos(4) .* inset.position(4) .* ARfig;
    ax_cb_phase.Position = [x_cb_phase, 0.05, 0.025*h, h_cb_phase*2];

    colormap(ax_cb_phase, inset.cmap)                          % Set colormap
    set(ax_cb_phase, 'YColor', 'k')
    set(ax_cb_phase, 'YTick', [1 256])
    set(ax_cb_phase, 'YTickLabel', ["\color[rgb]{1,1,1}0" "\color[rgb]{1,1,1}2\pi"])
    set(ax_cb_phase, 'YAxisLocation', 'right')
    set(ax_cb_phase, 'YDir', 'normal')
    set(ax_cb_phase, 'XTick', [])
    set(ax_cb_phase, 'FontSize', 12)
    set(ax_cb_phase, 'LineWidth', 0.01)
    drawnow


    % === RT === %
    % RT top
    s12.titlestr = "b. RT top correction (ours)";
    s12.filename = sprintf("tube-500nL-zoom8-zstep0.500um-top-RT-1_00001%s.fig", suffix);
    s12.pattern_rad = load_RT_pattern(fullfile(dirs.expdata, 'raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-top-λ808.0nm.mat'));
    s12.savedir = saveprops.dir;
    s12.position = [0.3-w/2, 0.75-h/2, w, h];
    ax12 = place_subplot(fig, s12, global_props);
    place_slm_inset(fig, ax12, s12, inset)

    % RT center
    s13.titlestr = "c. RT center correction (ours)";
    s13.filename = sprintf("tube-500nL-zoom8-zstep0.500um-center-RT-1_00001%s.fig", suffix);
    s13.pattern_rad = load_RT_pattern(fullfile(dirs.expdata, 'raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-center-λ808.0nm.mat'));
    s13.savedir = saveprops.dir;
    s13.position = [0.5-w/2, 0.75-h/2, w, h];
    ax13 = place_subplot(fig, s13, global_props);
    place_slm_inset(fig, ax13, s13, inset)

    % RT bottom
    s14.titlestr = "d. RT bottom correction (ours)";
    s14.filename = sprintf("tube-500nL-zoom8-zstep0.500um-bottom-RT-1_00001%s.fig", suffix);
    s14.pattern_rad = load_RT_pattern(fullfile(dirs.expdata, 'raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-bottom-λ808.0nm.mat'));
    s14.savedir = saveprops.dir;
    s14.position = [0.7-w/2, 0.75-h/2, w, h];
    ax14 = place_subplot(fig, s14, global_props);
    place_slm_inset(fig, ax14, s14, inset)

    % RT side
    s15.titlestr = "e. RT side correction (ours)";
    s15.filename = sprintf("tube-500nL-zoom8-zstep0.500um-side-RT-1_00001%s.fig", suffix);
    s15.pattern_rad = load_RT_pattern(fullfile(dirs.expdata, 'raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-side-λ808.0nm.mat'));
    s15.savedir = saveprops.dir;
    s15.position = [0.9-w/2, 0.75-h/2, w, h];
    ax15 = place_subplot(fig, s15, global_props);
    place_slm_inset(fig, ax15, s15, inset)

    % === AO === %
    % AO top
    s22.titlestr = "g. Zernike AO top correction";
    s22.filename = sprintf("tube-500nL-zoom8-zstep0.500um-top-AO-1_00001%s.fig", suffix);
    s22.pattern_rad = load_AO_pattern(fullfile(dirs.expdata, ao_dir, '30-Sep-2023-tube-500nL-top/tube_ao_739159.554324_tube-500nL-top/tube_ao_739159.554324_tube-500nL-top_optimal_pattern.mat'));
    s22.savedir = saveprops.dir;
    s22.position = [0.3-w/2, 0.25-h/2, w, h];
    ax22 = place_subplot(fig, s22, global_props);
    place_slm_inset(fig, ax22, s22, inset)

    % AO center
    s23.titlestr = "h. Zernike AO center correction";
    s23.filename = sprintf("tube-500nL-zoom8-zstep0.500um-center-AO-1_00001%s.fig", suffix);
    s23.pattern_rad = load_AO_pattern(fullfile(dirs.expdata, ao_dir, '30-Sep-2023-tube-500nL-center/tube_ao_739159.681107_tube-500nL-center/tube_ao_739159.681107_tube-500nL-center_optimal_pattern.mat'));
    s23.savedir = saveprops.dir;
    s23.position = [0.5-w/2, 0.25-h/2, w, h];
    ax23 = place_subplot(fig, s23, global_props);
    place_slm_inset(fig, ax23, s23, inset)

    % AO bottom
    s24.titlestr = "i. Zernike AO bottom correction";
    s24.filename = sprintf("tube-500nL-zoom8-zstep0.500um-bottom-AO-1_00001%s.fig", suffix);
    s24.pattern_rad = load_AO_pattern(fullfile(dirs.expdata, ao_dir, '30-Sep-2023-tube-500nL-bottom/tube_ao_739159.638038_tube-500nL-bottom/tube_ao_739159.638038_tube-500nL-bottom_optimal_pattern.mat'));
    s24.savedir = saveprops.dir;
    s24.position = [0.7-w/2, 0.25-h/2, w, h];
    ax24 = place_subplot(fig, s24, global_props);
    place_slm_inset(fig, ax24, s24, inset)

    % AO side
    s25.titlestr = "j. Zernike AO side correction";
    s25.filename = sprintf("tube-500nL-zoom8-zstep0.500um-side-AO-1_00001%s.fig", suffix);
    s25.pattern_rad = load_AO_pattern(fullfile(dirs.expdata, ao_dir, '30-Sep-2023-tube-500nL-side/tube_ao_739159.729111_tube-500nL-side/tube_ao_739159.729111_tube-500nL-side_optimal_pattern.mat'));
    s25.savedir = saveprops.dir;
    s25.position = [0.9-w/2, 0.25-h/2, w, h];
    ax25 = place_subplot(fig, s25, global_props);
    place_slm_inset(fig, ax25, s25, inset)

    movegui('center')

    % Save composition
    if saveprops.do_save
        savepath = fullfile(saveprops.dir, strcat(saveprops.save_prefix, suffix, '.pdf'));
        save_figtopdf(savepath);
        fprintf('Saved as: %s\n', savepath)
    end
end


% Plot subplot
function ax_infig = place_subplot(fig, s, global_props)
    filepath = fullfile(s.savedir, s.filename);
    subfig = openfig(filepath);
    ax_separate = subfig.findobj('Type', 'Axes');
    ax_infig = copyobj(ax_separate, fig);       % Copy axes from opened figure to desired figure
    close(subfig)                               % Close opened figure
    drawnow
    
    set(ax_infig, 'Position', s.position)

    % Title properties
    ax_infig.Title.String = s.titlestr;
    field_names = fields(global_props.title);
    for field_index = 1:numel(field_names)
        set(ax_infig.Title, field_names{field_index}, global_props.title.(field_names{field_index}));
    end
    
    % Set focus marker properties
    for child_index = 1:length(ax_infig.Children)
        child = ax_infig.Children(child_index);
        if isa(child, 'matlab.graphics.chart.primitive.Line')
            field_names = fields(global_props.focus_marker);
            for field_index = 1:numel(field_names)
                set(child, field_names{field_index}, global_props.focus_marker.(field_names{field_index}));
            end
        end
    end
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
    % Place figure inset with SLM pattern

    if nargin < 5
        insetshift = [0, 0];
    end
    pattern = mod(s.pattern_rad - s.pattern_rad(end/2, end/2), inset.modvalue);

    boxpos = plotboxpos(ax_subplot);                        % Get plot box position
%     annotation('rectangle', boxpos, 'edgecolor', 'r', 'linestyle', '-.');
    ARfig = fig.Position(4) ./ fig.Position(3);             % Aspect Ratio of figure
    xy = boxpos(1:2) + (inset.position(1:2) + insetshift) .* boxpos(4);    % x, y, relative
    wh = boxpos(4) .* inset.position(3:4) .* [ARfig 1];     % width, height, relative
    ax_inset = axes(fig, 'Position', [xy wh]);              % Apply inset position
    imagesc(ax_inset, pattern, 'AlphaData', inset.NA_mask, 'Tag', 'slm_pattern');
    axis(ax_inset, 'off')                                   % Hide axis
    ax_inset.CLim = inset.clim;                             % Set colormap limits
    colormap(ax_inset, inset.cmap)                          % Set colormap
    drawnow
end


function pattern_rad = load_RT_pattern(matpath)
    patterndata = load(matpath);
    pattern_rad = -patterndata.phase_SLM';
end


function pattern_rad = load_AO_pattern(matpath)
    patterndata = load(matpath);
    p = patterndata.p;

    % Generate uncalibrated zernike modes
    slm_pattern_size = size(patterndata.slm_pattern_2pi_optimal);
    coord_x = linspace(-1, 1, slm_pattern_size(1));
    coord_y = coord_x';
    Zcart_mode1 = zernfun_cart(coord_x, coord_y, p.mode1n, p.mode1m, p.truncate);
    Zcart_mode2 = zernfun_cart(coord_x, coord_y, p.mode2n, p.mode2m, p.truncate);

    % Compose AO pattern
    pattern_rad = patterndata.phase_amp_mode1_optimal .* Zcart_mode1 + patterndata.phase_amp_mode2_optimal .* Zcart_mode2;
end
