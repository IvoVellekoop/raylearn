clear; close all; clc

% Define found thicknesses
real_total_thickness_um = [[1   2   3   4   5] * 170,  400];
pred_total_thickness_um = [166 496 646 894 910      ,  362];

real_total_thickness_2nd_um = [1   2  3   4   5] * 170;
pred_total_thickness_2nd_um = [367 42 559 786 868];

% Subtract coverslip thickness for bottom objective
real_extra_thickness_um = real_total_thickness_um - 170;
pred_extra_thickness_um = pred_total_thickness_um - 170;
real_extra_thickness_2nd_um = real_total_thickness_2nd_um - 170;
pred_extra_thickness_2nd_um = pred_total_thickness_2nd_um - 170;

% Plot
um = ['(' char(181) 'm)'];
figure; hold on
thickness_plot     = plot(real_extra_thickness_um,     pred_extra_thickness_um,     'xb', MarkerSize=16);
thickness_2nd_plot = plot(real_extra_thickness_2nd_um, pred_extra_thickness_2nd_um, 'xr', MarkerSize=16);
lin = linspace(0, max(real_extra_thickness_um), 2);
lin_plot = plot(lin, lin, 'k');
hold off
xlabel(['Real extra thickness ' um])
ylabel(['extra thickness ' um])
legend([thickness_plot thickness_2nd_plot lin_plot], {'Predicted', 'Predicted 2nd round', 'Equality'}, Location='SouthEast')