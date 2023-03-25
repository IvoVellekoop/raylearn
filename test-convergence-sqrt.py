import torch
from matplotlib import pyplot as plt

from math_functions import sqrt_zero


beta = 100
A = torch.tensor(10., requires_grad=True)
B = torch.randn((4, 5))


def forward(B, A, beta):
    b = sqrt_zero(A + B, beta=beta)
    err = b.pow(2).sum() * 0.1
    return err, b


params = [A]
optimizer = torch.optim.Adam(params=params, lr=0.1, amsgrad=True, eps=1e-6)

As = []
errs = []
nans = []

# plt.figure

for t in range(200):
    err, b = forward(B, A, beta)

    As.append(A.detach().item())
    errs.append(err.detach().item())
    nans.append(b.isnan().sum().detach().item())

    # plt.imshow(b.detach(), vmin=0, vmax=4)
    # plt.title(f'iteration {t}')

    err.backward()
    # torch.nn.utils.clip_grad_norm_(params)
    optimizer.step()
    optimizer.zero_grad()

    # plt.draw()
# plt.close()


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
ax[0, 0].plot(errs, '-r', label='error')
ax[0, 0].plot(As, '-', label='parameter A')
ax[0, 0].plot(nans, '.-k', label='NaNs')
ax[0, 0].legend()
ax[0, 0].set_xlabel('iteration')

x = torch.linspace(-1, 1, 2000, requires_grad=True)
y = sqrt_zero(x, beta=beta)
ax[0, 1].plot(x.detach(), y.detach())
ax[0, 1].set_title(f'relu sqrt(x) for beta={beta:.3g}')
ax[0, 1].set_xlabel('x')
ax[0, 1].set_ylabel('relu sqrt(x)')

ones = torch.ones_like(y)
dydx = torch.autograd.grad(y, x, grad_outputs=ones, create_graph=True)[0]
ax[1, 0].plot(x.detach(), dydx.detach())
ax[1, 0].set_title(f'Gradient of relu sqrt(x) for beta={beta:.3g}')
ax[1, 0].set_xlabel('x')
ax[1, 0].set_ylabel('d(relu sqrt)/dx')
ax[1, 0].set_title(f'Derivative relu sqrt for beta={beta:.3g}')

ax[1, 1].plot(x.detach(), y.detach())
ax[1, 1].set_title(f'relu sqrt(x) for beta={beta:.3g}')
ax[1, 1].set_xlabel('x')
ax[1, 1].set_ylabel('relu sqrt(x)')
ax[1, 1].set_xlim((-0.1, 0.1))

fig.suptitle(f'beta={beta:.3g}')
plt.show()
