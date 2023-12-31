close all;

bottom_scan = load('/mnt/bmpi/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/adaptive-optics/27-Sep-2023-tube500nL-bottom-scalescan/tube_scale_scan_739156.565932_tube500nL-bottom-scalescan/tube_scale_scan_739156.565932_tube500nL-bottom-scalescan_angle_scan.mat');
center_scan = load('/mnt/bmpi/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/adaptive-optics/27-Sep-2023-tube500nL-center-scalescan/tube_scale_scan_739156.610481_tube500nL-center-scalescan/tube_scale_scan_739156.610481_tube500nL-center-scalescan_angle_scan.mat');
top_scan = load('/mnt/bmpi/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/adaptive-optics/27-Sep-2023-tube500nL-top-scalescan/tube_scale_scan_739156.652694_tube500nL-top-scalescan/tube_scale_scan_739156.652694_tube500nL-top-scalescan_angle_scan.mat');
side_scan = load('/mnt/bmpi/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/adaptive-optics/27-Sep-2023-tube500nL-side-scalescan/tube_scale_scan_739156.696890_tube500nL-side-scalescan/tube_scale_scan_739156.696890_tube500nL-side-scalescan_angle_scan.mat');
clc;

figure; gca; hold on
plot_scan(bottom_scan);
plot_scan(center_scan);
plot_scan(top_scan);
plot_scan(side_scan);
plot_scan_flat(bottom_scan);
plot_scan_flat(center_scan);
plot_scan_flat(top_scan);
plot_scan_flat(side_scan);
hold off
legend('Location', 'NorthWest')
xlabel('SLM scale')
ylabel('signal \sigma')
title('Effect of scaling the pattern')
grid on
set(gca, 'fontsize', 14)

figure; gca; hold on
plot_scan_contrast(bottom_scan);
plot_scan_contrast(center_scan);
plot_scan_contrast(top_scan);
plot_scan_contrast(side_scan);
hold off
legend('Location', 'NorthWest')
xlabel('SLM scale')
ylabel('contrast enhancement \eta_\sigma')
title('Effect of scaling the pattern')
grid on
set(gca, 'fontsize', 14)

function plot_scan(scan)
    plot(scan.p.scale_range, scan.all_signal_std_corrected, '.-', 'DisplayName', scan.p.samplename(11:end-10))
end

function plot_scan_flat(scan)
    plot(scan.p.scale_range, scan.all_signal_std_flat, '.-', 'DisplayName', [scan.p.samplename(11:end-10) ' flat'])
end

function plot_scan_contrast(scan)
    plot(scan.p.scale_range, scan.all_signal_std_corrected ./ median(scan.all_signal_std_flat), '.-', 'DisplayName', scan.p.samplename(11:end-10))
end
