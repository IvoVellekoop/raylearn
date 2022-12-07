"""
Learn the parameters of an optical system.
"""

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from plot_functions import plot_coords, format_prefix


torch.set_default_tensor_type('torch.DoubleTensor')


class System():
    """
    System.

    An object with properties that can be optimized, by running an optimize method.
    """
    def __init__(self):
        super().__init__()
        self.parameters = {}

    def optimize(self):
        # Define optimizer
        optimizer = torch.optim.Adam([
                {'lr': 1.0e-1, 'params': self.parameters['angle'].values()},
                {'lr': 1.0e-4, 'params': self.parameters['obj'].values()},
                {'lr': 2.0e-3, 'params': self.parameters['other'].values()},
            ], lr=1.0e-5)

        iterations = 2
        errors = torch.zeros(iterations)

        # Initialize logs for tracking each parameter
        self.parameters_logs = {}
        for groupname in self.parameters:
            self.parameters_logs[groupname] = {}
            for paramname in self.parameters[groupname]:
                self.parameters_logs[groupname][paramname] = torch.zeros(iterations)

        trange = tqdm(range(iterations), desc='error: -')


        # Plot
        if do_plot_coverslip:
            fig, ax = plt.subplots(nrows=2, figsize=(5, 10), dpi=110)

            fig_tpm = plt.figure(figsize=(15, 4), dpi=110)
            ax_tpm = plt.gca()


        for t in trange:
            # === Learn === #
            # Forward pass
            tpm.update()
            cam_ft_coords, cam_im_coords = tpm.raytrace()

            # Compute and print error
            error = MSE(cam_ft_coords_gt, cam_ft_coords) \
                + MSE(cam_im_coords_gt, cam_im_coords) \

            error_value = error.detach().item()
            errors[t] = error_value

            for groupname in self.parameters:
                for paramname in self.parameters[groupname]:
                    self.parameters_logs[groupname][paramname][t] = params_obj1_zshift[groupname][paramname].detach().item()

            trange.desc = f'error: {error_value:<8.3g}' \
                + f'coverslip thickness: {format_prefix(tpm.total_coverslip_thickness, "8.3f")}m'

            # error.backward(retain_graph=True)
            error.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Plot
            if t % 200 == 0 and do_plot_coverslip:
                plt.figure(fig.number)

                # Fourier cam
                cam_ft_coord_pairs_x, cam_ft_coord_pairs_y = \
                        torch.stack((cam_ft_coords_gt, cam_ft_coords)).detach().unbind(-1)

                ax[0].clear()
                ax[0].plot(cam_ft_coord_pairs_x.view(2, -1), cam_ft_coord_pairs_y.view(2, -1),
                        color='lightgrey')
                plot_coords(ax[0], cam_ft_coords_gt[:, :, :], {'label': 'measured'})
                plot_coords(ax[0], cam_ft_coords[:, :, :], {'label': 'sim'})

                ax[0].set_ylabel('y (pix)')
                ax[0].set_xlabel('x (pix)')
                ax[0].legend(loc=1)
                ax[0].set_title(f'Fourier Cam | coverslip={format_prefix(tpm.total_coverslip_thickness)}m | iter: {t}')

                # Image cam
                cam_im_coord_pairs_x, cam_im_coord_pairs_y = \
                    torch.stack((cam_im_coords_gt, cam_im_coords)).detach().unbind(-1)

                ax[1].clear()
                ax[1].plot(cam_im_coord_pairs_x.view(2, -1), cam_im_coord_pairs_y.view(2, -1),
                        color='lightgrey')
                plot_coords(ax[1], cam_im_coords_gt[:, :, :], {'label': 'measured'})
                plot_coords(ax[1], cam_im_coords[:, :, :], {'label': 'sim'})

                ax[1].set_ylabel('y (pix)')
                ax[1].set_xlabel('x (pix)')
                ax[1].legend(loc=1)
                ax[1].set_title(f'Image Cam | iter: {t}')

                plt.draw()
                plt.pause(1e-3)

                plt.figure(fig_tpm.number)
                ax_tpm.clear()
                tpm.plot(ax_tpm, fraction=0.01)
                plt.draw()
                plt.pause(1e-3)

        for groupname in self.parameters:
            print('\n' + groupname + ':')
            for paramname in self.parameters[groupname]:
                if groupname == 'angle':
                    print(f'  {paramname}: {self.parameters[groupname][paramname].detach().item():.3f}rad')
                else:
                    print(f'  {paramname}: {format_prefix(self.parameters[groupname][paramname], ".3f")}m')


        if do_plot_coverslip:
            fig, ax1 = plt.subplots(figsize=(7, 7))
            fig.dpi = 144

            # Plot error
            errorcolor = 'tab:red'
            RMSEs = np.sqrt(errors.detach().cpu())
            ax1.plot(RMSEs, label='error', color=errorcolor)
            ax1.set_ylabel('Error (pix)')
            ax1.set_ylim((0, max(RMSEs)))
            ax1.legend(loc=2)
            ax1.legend()

            ax2 = ax1.twinx()
            for groupname in self.parameters:
                for paramname in self.parameters_logs[groupname]:
                    ax2.plot(self.parameters_logs[groupname][paramname], label=paramname)
            ax2.set_ylabel('Parameter (m | rad)')
            ax2.legend(loc=1)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.title('Learning parameters')
            plt.show()
