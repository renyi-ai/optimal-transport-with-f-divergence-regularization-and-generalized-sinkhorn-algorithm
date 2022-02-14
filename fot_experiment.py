from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import argparse
import datetime
import imageio
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time


def draw_samples(fname, n):
    img = imageio.imread(fname, as_gray=True)
    img = (img[::-1, :]) / 255.0
    A = 1 - img
    xg, yg = np.meshgrid(np.linspace(0, 1, A.shape[0]), np.linspace(0, 1, A.shape[1]))
    grid = list(zip(xg.ravel(), yg.ravel()))
    dens = A.ravel() / A.sum()
    dots = np.array(random.choices(grid, dens, k=n))
    dots += 0.005 * np.random.standard_normal(dots.shape)
    return torch.from_numpy(dots).float()


def display_samples(ax, x, color):
    x_ = x.data.cpu().numpy()
    ax.scatter(x_[:, 0], x_[:, 1], 16 * 500 / len(x_), color)


class GammaSteps:
    counter = 0


class GammaKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f_x):
        f_x = f_x.detach()
        ctx.save_for_backward(f_x)
        return torch.log(torch.mean(torch.exp(f_x), dim=-1, keepdim=True))

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            f_x, = ctx.saved_tensors
            grad_input_f = torch.exp(f_x) / torch.sum(torch.exp(f_x), dim=-1, keepdim=True)
        return grad_input_f * grad_output


class GammaReverseKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f_x):
        f_x = f_x.detach()
        gamma = torch.max(f_x, dim=-1, keepdim=True)[0] - 1 + 1e-4
        GammaSteps.counter = 0
        for _ in range(10000):
            F_gamma = -torch.mean(1 / (1 - f_x + gamma), dim=-1, keepdim=True) + 1
            F_gamma_derivative = torch.mean(1 / ((1 - f_x + gamma) ** 2), dim=-1, keepdim=True)
            step = F_gamma / F_gamma_derivative
            gamma = gamma - step
            GammaSteps.counter = GammaSteps.counter + 1
            if torch.mean(step ** 2).item() < 1e-12:
                break
        ctx.save_for_backward(f_x, gamma)
        return gamma

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            f_x, gamma = ctx.saved_tensors
            F_gamma_gradient = -(1 / float(f_x.size(-1))) * (1 / ((1 - f_x + gamma) ** 2))
            F_gamma_derivative = torch.mean(1 / ((1 - f_x + gamma) ** 2), dim=-1, keepdim=True)
            grad_input_f = -F_gamma_gradient / F_gamma_derivative
        return grad_input_f * grad_output


class GammaChi2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f_x):
        f_x = f_x.detach()
        gamma = torch.log(torch.mean(torch.exp(f_x), dim=-1, keepdim=True))
        GammaSteps.counter = 0
        for _ in range(10000):
            F_gamma = -torch.mean(((f_x - gamma) / 2 + 1) * (f_x - gamma >= -2).to(f_x.dtype), dim=-1, keepdim=True) + 1
            F_gamma_derivative = torch.mean((f_x - gamma >= -2).to(f_x.dtype) / 2, dim=-1, keepdim=True)
            step = F_gamma / F_gamma_derivative
            gamma = gamma - step
            GammaSteps.counter = GammaSteps.counter + 1
            if torch.mean(step ** 2).item() < 1e-6:
                break
        ctx.save_for_backward(f_x, gamma)
        return gamma

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            f_x, gamma = ctx.saved_tensors
            F_gamma_gradient = -(1 / float(f_x.size(-1))) * ((f_x - gamma >= -2).to(f_x.dtype) / 2)
            F_gamma_derivative = torch.mean((f_x - gamma >= -2).to(f_x.dtype) / 2, dim=-1, keepdim=True)
            grad_input_f = -F_gamma_gradient / F_gamma_derivative
        return grad_input_f * grad_output


class GammaReverseChi2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f_x):
        f_x = f_x.detach()
        gamma = torch.max(f_x, dim=-1, keepdim=True)[0] - 1 + 1e-6
        GammaSteps.counter = 0
        for _ in range(10000):
            F_gamma = -torch.mean(1 / torch.sqrt(1 - f_x + gamma), dim=-1, keepdim=True) + 1
            F_gamma_derivative = torch.mean(1 / (2 * (torch.sqrt(1 - f_x + gamma) ** 3)), dim=-1, keepdim=True)
            step = F_gamma / F_gamma_derivative
            gamma = gamma - step
            GammaSteps.counter = GammaSteps.counter + 1
            if torch.mean(step ** 2).item() < 1e-12:
                break
        ctx.save_for_backward(f_x, gamma)
        return gamma

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            f_x, gamma = ctx.saved_tensors
            F_gamma_gradient = -(1 / float(f_x.size(-1))) * (1 / (2 * (torch.sqrt(1 - f_x + gamma) ** 3)))
            F_gamma_derivative = torch.mean(1 / (2 * (torch.sqrt(1 - f_x + gamma) ** 3)), dim=-1, keepdim=True)
            grad_input_f = -F_gamma_gradient / F_gamma_derivative
        return grad_input_f * grad_output


class GammaHellinger2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f_x):
        f_x = f_x.detach()
        gamma = torch.max(f_x, dim=-1, keepdim=True)[0] - 1 + 1e-2
        GammaSteps.counter = 0
        for _ in range(10000):
            F_gamma = -torch.mean(1 / ((1 - f_x + gamma) ** 2), dim=-1, keepdim=True) + 1
            F_gamma_derivative = torch.mean(2 / ((1 - f_x + gamma) ** 3), dim=-1, keepdim=True)
            step = F_gamma / F_gamma_derivative
            gamma = gamma - step
            GammaSteps.counter = GammaSteps.counter + 1
            if torch.mean(step ** 2).item() < 1e-12:
                break
        ctx.save_for_backward(f_x, gamma)
        return gamma

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            f_x, gamma = ctx.saved_tensors
            F_gamma_gradient = -(1 / float(f_x.size(-1))) * (2 / ((1 - f_x + gamma) ** 3))
            F_gamma_derivative = torch.mean(2 / ((1 - f_x + gamma) ** 3), dim=-1, keepdim=True)
            grad_input_f = -F_gamma_gradient / F_gamma_derivative
        return grad_input_f * grad_output


class GammaJS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f_x):
        f_x = f_x.detach()
        gamma = torch.max(f_x, dim=-1, keepdim=True)[0] - math.log(2) + 1e-3
        GammaSteps.counter = 0
        for _ in range(10000):
            F_gamma = -torch.mean(1 / (2 * torch.exp(gamma - f_x) - 1), dim=-1, keepdim=True) + 1
            F_gamma_derivative = torch.mean((2 * torch.exp(f_x - gamma)) / ((torch.exp(f_x - gamma) - 2) ** 2), dim=-1, keepdim=True)
            step = F_gamma / F_gamma_derivative
            gamma = gamma - step
            GammaSteps.counter = GammaSteps.counter + 1
            if torch.mean(step ** 2).item() < 1e-12:
                break
        ctx.save_for_backward(f_x, gamma)
        return gamma

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            f_x, gamma = ctx.saved_tensors
            F_gamma_gradient = -(1 / float(f_x.size(-1))) * (
                (2 * torch.exp(f_x - gamma)) / ((torch.exp(f_x - gamma) - 2) ** 2)
            )
            F_gamma_derivative = torch.mean((2 * torch.exp(f_x - gamma)) / ((torch.exp(f_x - gamma) - 2) ** 2), dim=-1, keepdim=True)
            grad_input_f = -F_gamma_gradient / F_gamma_derivative
        return grad_input_f * grad_output


class LambertW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        z = z.detach()
        w = torch.log(1 + z)
        for i in range(10000):
            step = (w * torch.exp(w) - z) / (torch.exp(w) + w * torch.exp(w))
            w = w - step
            if torch.mean(step ** 2).item() < 1e-6:
                break
        ctx.save_for_backward(z, w)
        return w

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            z, w = ctx.saved_tensors
            grad_input_z = w / (z * (1 + w))
        return grad_input_z * grad_output


class LambertWCircExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.detach()
        y = torch.clamp_min(1 - x, min=1)
        for i in range(10000):
            step = (y - torch.exp(1 - x - y)) / (1 + y)
            y = y - step
            if torch.mean(step ** 2).item() < 1e-6:
                break
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            x, y = ctx.saved_tensors
            grad_input_x = -torch.exp(1 - x - y) / (1 + y)
        return grad_input_x * grad_output


class GammaJeffreys(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f_x):
        f_x = f_x.detach()
        gamma = torch.log(torch.mean(torch.exp(f_x), dim=-1, keepdim=True))
        GammaSteps.counter = 0
        for i in range(10000):
            # w = LambertW.apply(torch.exp(1 - f_x + gamma))
            w = LambertWCircExp.apply(f_x - gamma)
            F_gamma = -torch.mean(1 / w, dim=-1, keepdim=True) + 1
            F_gamma_derivative = torch.mean((1 / w) - (1 / (w + 1)), dim=-1, keepdim=True)
            step = F_gamma / F_gamma_derivative
            gamma = gamma - step
            GammaSteps.counter = GammaSteps.counter + 1
            if torch.mean(step ** 2).item() < 1e-6:
                break
        ctx.save_for_backward(f_x, gamma)
        return gamma

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            f_x, gamma = ctx.saved_tensors
            w = LambertW.apply(torch.exp(1 - f_x + gamma))
            F_gamma_gradient = -(1 / float(f_x.size(-1))) * ((1 / w) - (1 / (w + 1)))
            F_gamma_derivative = torch.mean((1 / w) - (1 / (w + 1)), dim=-1, keepdim=True)
            grad_input_f = -F_gamma_gradient / F_gamma_derivative
        return grad_input_f * grad_output


class GammaTriangular(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f_x):
        f_x = f_x.detach()
        gamma = torch.max(f_x, dim=-1, keepdim=True)[0] - 1 + 1e-5
        GammaSteps.counter = 0
        for _ in range(10000):
            F_gamma = -torch.mean((2 / ((1 - f_x + gamma) ** (1/2)) - 1) * (f_x - gamma >= -3).to(f_x.dtype), dim=-1, keepdim=True) + 1
            F_gamma_derivative = torch.mean((1 / (((1 - f_x + gamma) ** (1/2)) ** 3)) * (f_x - gamma >= -3).to(f_x.dtype), dim=-1, keepdim=True)
            step = F_gamma / F_gamma_derivative
            gamma = gamma - step
            GammaSteps.counter = GammaSteps.counter + 1
            if torch.mean(step ** 2).item() < 1e-12:
                break
        ctx.save_for_backward(f_x, gamma)
        return gamma

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            f_x, gamma = ctx.saved_tensors
            F_gamma_gradient = -(1 / float(f_x.size(-1))) * (
                1 / (((1 - f_x + gamma) ** (1/2)) ** 3)
            ) * (f_x - gamma >= -3).float()
            F_gamma_derivative = torch.mean((1 / (((1 - f_x + gamma) ** (1/2)) ** 3)) * (f_x - gamma >= -3).to(f_x.dtype), dim=-1, keepdim=True)
            grad_input_f = -F_gamma_gradient / F_gamma_derivative
        return grad_input_f * grad_output


def stabilized(fn):
    stabilized_fn = lambda f_x: (
        fn(f_x - torch.max(f_x.detach(), dim=-1, keepdim=True)[0]) + torch.max(f_x.detach(), dim=-1)[0]
    )
    return stabilized_fn


phi_conjugate_derivatives = {
    'kl': lambda x: torch.exp(x),
    'rkl': lambda x: torch.ones_like(x) / (torch.ones_like(x) - x),
    'chi2': lambda x: (x >= -2).to(x.dtype) * (x / 2 + 1),
    'rchi2': lambda x: torch.ones_like(x) / torch.sqrt(torch.ones_like(x) - x),
    'hellinger2': lambda x: torch.ones_like(x) / ((torch.ones_like(x) - x) ** 2),
    'js': lambda x: torch.ones_like(x) / (2 * torch.exp(-x) - torch.ones_like(x)),
    # 'jeffreys': lambda x: torch.ones_like(x) / LambertW.apply(torch.exp(torch.ones_like(x) - x)),
    'jeffreys': lambda x: torch.ones_like(x) / LambertWCircExp.apply(x),
    'triangular': lambda x: (x >= -3).to(x.dtype) * ((2 * torch.ones_like(x)) / torch.sqrt(torch.ones_like(x) - x) - 1),
}


gamma_fns = {
    'kl': lambda f_x: GammaKL.apply(f_x).squeeze(-1),
    'rkl': lambda f_x: GammaReverseKL.apply(f_x).squeeze(-1),
    'chi2': lambda f_x: GammaChi2.apply(f_x).squeeze(-1),
    'rchi2': lambda f_x: GammaReverseChi2.apply(f_x).squeeze(-1),
    'hellinger2': lambda f_x: GammaHellinger2.apply(f_x).squeeze(-1),
    'js': lambda f_x: GammaJS.apply(f_x).squeeze(-1),
    'jeffreys': lambda f_x: GammaJeffreys.apply(f_x).squeeze(-1),
    'triangular': lambda f_x: GammaTriangular.apply(f_x).squeeze(-1),
}

def get_potentials(eps, divergence, c, tol):
    gamma_fn = gamma_fns[divergence]
    stabilized_gamma_fn = stabilized(gamma_fn)
    with torch.no_grad():
        f = torch.zeros(c.size(1), device=c.device)
        steps = 0
        while True:
            f_prev = f
            g = -eps * stabilized_gamma_fn((1 / eps) * (f.detach().view(1, -1) - c))
            f = -eps * stabilized_gamma_fn((1 / eps) * (g.detach().view(1, -1) - c.t()))
            step = torch.abs((f - f[0]) - (f_prev - f_prev[0])).max().item()
            print(f'get_potentials step {step:.8f}', end="\r")
            steps += 1
            if step < tol:
                break
        print(f'get_potentials done in {steps} steps')
    return f, g


def get_joint_distribution(f, g, eps, divergence, c):
    phi_conjugate_derivative = phi_conjugate_derivatives[divergence]
    gamma_fn = gamma_fns[divergence]
    stabilized_gamma_fn = stabilized(gamma_fn)
    f, g, c = (f.double(), g.double(), c.double())
    g = -eps * stabilized_gamma_fn((1 / eps) * (f.view(1, -1) - c))
    f = -eps * stabilized_gamma_fn((1 / eps) * (g.view(1, -1) - c.t()))
    mu_times_nu = torch.ones_like(c) / (c.size(0) * c.size(1))
    pi = phi_conjugate_derivative((1 / eps) * ((f.view(1, -1) + g.view(-1, 1)) - c)).float() * mu_times_nu
    return pi

def round_coupling(pi,f,g):

    r = torch.ones_like(g) / g.size(0)
    c = torch.ones_like(f) / f.size(0)

    mu = torch.mv(pi, torch.ones_like(f))
    x = torch.diag(torch.minimum(torch.nan_to_num(r / mu, nan=1), torch.ones_like(g)))
    f_prime = torch.matmul(x, pi)

    nu = torch.mv(f_prime.t(), torch.ones_like(g))
    y = torch.diag(torch.minimum(torch.nan_to_num(c / nu, nan=1), torch.ones_like(f)))
    f_prime_prime = torch.matmul(f_prime, y)

    err_r = r - torch.mv(f_prime_prime, torch.ones_like(f))
    err_c = c - torch.mv(f_prime_prime.t(), torch.ones_like(g))

    l_1_err_r = torch.sum(torch.abs(err_r))

    correction_matrix = torch.outer(err_r, err_c) / l_1_err_r

    return f_prime_prime + correction_matrix


def run_experiment(args):
    print(f'run_experiment {args}')

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    datasets = {
        "moons": ("moon_a.png", "moon_b.png"),
        "densities": ("density_a.png", "density_b.png"),
        "slopes": ("slope_a.png", "slope_b.png"),
        "crescents": ("crescent_a.png", "crescent_b.png"),
    }
    if args.dataset == "crescents":
        ax_limits = [0, 1, 0, 1]
    else:
        ax_limits = [0, 1, 0.125, 0.875]

    timestamp = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")
    dir_name = args.log_dir + f'/{args.dataset}_mu_{args.mu_size}_nu_{args.nu_size}_' \
                              f'{args.divergence}_eps_{args.epsilon}_' \
                              f'tol_{args.tolerance}_rs{args.random_seed}_' \
                              f'{("double" if args.double else "single")}_{timestamp}/'
    os.makedirs(os.path.dirname(dir_name), exist_ok=True)
    writer = SummaryWriter(log_dir=dir_name)

    mu = draw_samples(args.data_dir + datasets[args.dataset][0], args.mu_size).cuda()
    nu = draw_samples(args.data_dir + datasets[args.dataset][1], args.nu_size).cuda()
    if args.double:
        mu = mu.double()
        nu = nu.double()
    mu.requires_grad_(True)
    eps = args.epsilon
    divergence = args.divergence
    c_fn = lambda x_, y_: torch.sum((x_.view(1, -1, 2) - y_.view(-1, 1, 2)) ** 2, dim=-1) / 2
    tol = args.tolerance

    c = c_fn(mu, nu)
    if args.double:
        c = c.double()

    gamma_fn = gamma_fns[divergence]
    stabilized_gamma_fn = stabilized(gamma_fn)
    phi_conjugate_derivative = phi_conjugate_derivatives[divergence]
    mu_times_nu = torch.ones_like(c) / (c.size(0) * c.size(1))
    total_time = 0
    steps = 0
    with torch.no_grad():
        f = torch.zeros(c.size(1), device=c.device)
        while True:
            f_prev = f
            before = time.time()
            g = -eps * stabilized_gamma_fn((1 / eps) * (f.detach().view(1, -1) - c))
            gamma_steps_1 = GammaSteps.counter
            after = time.time()
            c_transform_1_time = after - before
            before = time.time()
            f = -eps * stabilized_gamma_fn((1 / eps) * (g.detach().view(1, -1) - c.t()))
            gamma_steps_2 = GammaSteps.counter
            after = time.time()
            c_transform_2_time = after - before
            total_time += c_transform_1_time + c_transform_2_time
            diffs = torch.abs((f - f[0]) - (f_prev - f_prev[0]))
            max_diff = diffs.max().item()
            print(f'get_potentials diff {max_diff:.8f}', end="\r")
            pi = phi_conjugate_derivative((1 / eps) * ((f.view(1, -1) + g.view(-1, 1)) - c)) * mu_times_nu
            cost = torch.sum(pi.detach() * c)
            marginal_error_1 = torch.abs(torch.mv(pi, torch.ones_like(f)) - (torch.ones_like(g) / g.size(0))).sum()
            marginal_error_2 = torch.abs(torch.mv(pi.t(), torch.ones_like(g)) - (torch.ones_like(f) / f.size(0))).sum()
            steps += 1
            writer.add_scalar('timing/c_transform_1_time', c_transform_1_time, global_step=steps)
            writer.add_scalar('timing/c_transform_2_time', c_transform_2_time, global_step=steps)
            writer.add_scalar('timing/total_time', total_time, global_step=steps)
            writer.add_scalar('timing/gamma_steps_1', gamma_steps_1, global_step=steps)
            writer.add_scalar('timing/gamma_steps_2', gamma_steps_2, global_step=steps)
            writer.add_scalar('max_diff', max_diff, global_step=steps)
            writer.add_scalar('cost', cost, global_step=steps)
            writer.add_scalar('cost_per_time', cost, global_step=int(total_time))
            writer.add_scalar('marginal_error_1', marginal_error_1, global_step=steps)
            writer.add_scalar('marginal_error_2', marginal_error_2, global_step=steps)
            writer.add_histogram('diffs', diffs.view(-1).detach().cpu().numpy(), global_step=steps)
            writer.add_histogram('pi', pi.view(-1).detach().cpu().numpy(), global_step=steps)
            if max_diff < tol or max_diff != max_diff:
                break
        print(f'get_potentials done in {steps} steps')

    pi = phi_conjugate_derivative((1 / eps) * ((f.view(1, -1) + g.view(-1, 1)) - c)) * mu_times_nu
    cost = torch.sum(pi.detach() * c)
    marginal_error_1 = torch.abs(torch.mv(pi, torch.ones_like(f)) - (torch.ones_like(g) / g.size(0))).sum()
    marginal_error_2 = torch.abs(torch.mv(pi.t(), torch.ones_like(g)) - (torch.ones_like(f) / f.size(0))).sum()
    pi_nonzeros_ratio = (pi > 0).float().mean().item()
    writer.add_scalar('final/cost', cost.item())
    writer.add_scalar('final/steps', steps)
    writer.add_scalar('final/total_time', total_time)
    writer.add_scalar('final/marginal_error_1', marginal_error_1.item())
    writer.add_scalar('final/marginal_error_2', marginal_error_2.item())
    writer.add_scalar('final/pi_nonzeros_ratio', pi_nonzeros_ratio)

    pi_rounded = round_coupling(pi, f, g)
    cost_rounded = torch.sum(pi_rounded.detach() * c)
    marginal_error_1_rounded = torch.abs(torch.mv(pi_rounded, torch.ones_like(f)) - (torch.ones_like(g) / g.size(0))).sum()
    marginal_error_2_rounded = torch.abs(torch.mv(pi_rounded.t(), torch.ones_like(g)) - (torch.ones_like(f) / f.size(0))).sum()
    pi_nonzeros_ratio_rounded = (pi_rounded > 0).float().mean().item()
    writer.add_scalar('final_rounded/cost', cost_rounded.item())
    writer.add_scalar('final_rounded/marginal_error_1', marginal_error_1_rounded.item())
    writer.add_scalar('final_rounded/marginal_error_2', marginal_error_2_rounded.item())
    writer.add_scalar('final_rounded/pi_nonzeros_ratio', pi_nonzeros_ratio_rounded)

    loss = torch.sum(pi.detach() * c)
    loss.backward()

    plt.scatter([10], [10])
    display_samples(plt.gca(), nu, (0.55, 0.55, 0.95))
    display_samples(plt.gca(), mu.data - mu.grad * args.mu_size, (0.95, 0.55, 0.55))
    plt.axis("equal")
    plt.axis(ax_limits)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', adjustable='box')
    writer.add_figure('transport', plt.gcf(), global_step=steps)
    plt.clf()

    print("Done.")

    writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir",                default="logs/fot_experiment/")
    parser.add_argument("--data_dir",               default="data/fot_experiment/")
    parser.add_argument("--random_seed", type=int,   default=0)
    parser.add_argument("--mu_size",     type=int,   default=500)
    parser.add_argument("--nu_size",     type=int,   default=500)
    parser.add_argument("--tolerance",   type=float, default=1e-6)
    parser.add_argument("--epsilon",     type=float, default=1e-2)
    parser.add_argument("--dataset",    default="densities", choices=[
        "moons", "densities", "slopes", "crescents"
    ])
    parser.add_argument("--divergence", default="kl", choices=[
        "kl", "rkl", "chi2", "rchi2", "hellinger2", "js", "jeffreys", "triangular"
    ])
    parser.add_argument('--double',     dest='double',  action='store_true')
    parser.add_argument('--float',      dest='double',  action='store_false')
    parser.set_defaults(double=False)
    args = parser.parse_args()

    run_experiment(args)

