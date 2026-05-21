# "\n".join([l for l in torch.cuda.memory_summary(device="cuda:1",abbreviated=True).split("\n") if "Allocations" in l])
import types

import numpy as np
import qmcpy as qp


class DarcyFlow2d(object):
    """
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> qmean = 0
    >>> df = DarcyFlow2d()
    >>> for l in range(3):
    ...     x = rng.uniform(size=(2**9,df.d))
    ...     y = df.ml(level=l,samples=x)
    ...     ymean_l = y.mean()
    ...     ystd_l = y.std(ddof=1)
    ...     qmean += ymean_l
    ...     print("Qmean[l] = %-10.3f Ymean[l] = %-15.3e Ystd[l] = %.3e"%(qmean,ymean_l,ystd_l))
    Qmean[l] = 0.041      Ymean[l] = 4.105e-02       Ystd[l] = 7.782e-02
    Qmean[l] = 0.042      Ymean[l] = 8.549e-04       Ystd[l] = 6.738e-03
    Qmean[l] = 0.042      Ymean[l] = 3.260e-04       Ystd[l] = 1.940e-03
    """

    def __init__(
        self,
        levels=None,
        n_coarsest=None,
        nonlinearity_factor=None,
        f_kernel=None,
        f_kernel_lengthscales=None,
        f_kernel_scale=None,
        f_kmat_noise=None,
        u_kernel=None,
        u_kernel_lengthscales=None,
        u_kernel_scale=None,
        u_kmat_noise=None,
        device="cpu",
    ):
        allnone = (
            levels is None
            and n_coarsest is None
            and nonlinearity_factor is None
            and f_kernel is None
            and f_kernel_lengthscales is None
            and f_kernel_scale is None
            and f_kmat_noise is None
            and u_kernel is None
            and u_kernel_lengthscales is None
            and u_kernel_scale is None
            and u_kmat_noise is None
        )
        if allnone:
            # self.raw_costs = np.array([4.169996827840805e-05,9.962388128042222e-05,2.740689277648926e-03])
            self.raw_costs = np.array(
                [4.217835143208504e-05, 9.883953630924225e-05, 2.766717433929443e-03]
            )
            self.adjusted_costs = self.raw_costs / self.raw_costs[-1]
            self.exact = types.SimpleNamespace()
            self.exact.Q = types.SimpleNamespace()
            self.exact.Y = types.SimpleNamespace()
            # self.exact_values = np.array([0.041626055544059,0.013428061097414,0.012431490423060])
            # self.exact_diffs = np.array([4.162605554405886e-02,-2.819799444664468e-02,-9.965706743537035e-04])
            # self.exact_values = np.array([0.045731827083527,0.046956554309946,0.046893108081545])
            # self.exact_diffs = np.array([4.573182708352660e-02,1.224727226419649e-03,-6.344622840109143e-05])
            # self.exact.Q.mean = lambda level: self.exact_values[level]
            # self.exact.Y.mean = lambda level: self.exact_diffs[level]
        levels = 3 if levels is None else levels
        n_coarsest = 8 if n_coarsest is None else n_coarsest
        nonlinearity_factor = 1 if nonlinearity_factor is None else nonlinearity_factor
        f_kernel = qp.KernelMatern12 if f_kernel is None else f_kernel
        # f_kernel_lengthscales = 0.1  if f_kernel_lengthscales is None else f_kernel_lengthscales
        f_kernel_lengthscales = (
            0.5 if f_kernel_lengthscales is None else f_kernel_lengthscales
        )
        f_kernel_scale = 1.0 if f_kernel_scale is None else f_kernel_scale
        f_kmat_noise = 1e-5 if f_kmat_noise is None else f_kmat_noise
        u_kernel = qp.KernelGaussian if u_kernel is None else u_kernel
        # u_kernel_lengthscales = 0.1  if u_kernel_lengthscales is None else u_kernel_lengthscales
        u_kernel_lengthscales = (
            0.5 if u_kernel_lengthscales is None else u_kernel_lengthscales
        )
        u_kernel_scale = 1.0 if u_kernel_scale is None else u_kernel_scale
        u_kmat_noise = 1e-5 if u_kmat_noise is None else u_kmat_noise
        import torch

        assert torch.get_default_dtype() == torch.float64
        self.device = device
        self.levels = levels
        self.nonlinearity_factor = nonlinearity_factor
        assert n_coarsest > 0 and np.log2(n_coarsest) % 1 == 0
        self.ns = n_coarsest * 2 ** np.arange(self.levels)
        self.dx = 1 / self.ns
        self.ps = self.ns - 1
        self.pmax = self.ps[-1]
        self.p2s = self.ps**2
        self.p2max = self.p2s[-1]
        self.p2max_u = 3 * self.p2max
        self.d = int(self.p2max + self.p2max_u)
        x1 = torch.linspace(0, 1, self.ns[-1] + 1, device=self.device)[1:-1]
        x2 = torch.linspace(0, 1, self.ns[-1] + 1, device=self.device)[1:-1]
        x1mesh, x2mesh = torch.meshgrid(x1, x2, indexing="ij")
        xticks = torch.vstack([x1mesh.flatten(), x2mesh.flatten()]).T
        f_kernel = f_kernel(
            d=2,
            lengthscales=f_kernel_lengthscales,
            scale=f_kernel_scale,
            torchify=True,
            device=self.device,
        )
        kmat_f = f_kernel(
            xticks[:, None, :], xticks[None, :, :]
        ).detach() + f_kmat_noise * torch.eye(self.p2max, device=self.device)
        evals_f, evecs_f = torch.linalg.eigh(kmat_f)
        self.factor_f = evecs_f * torch.sqrt(evals_f)
        assert torch.allclose(self.factor_f @ self.factor_f.T, kmat_f, atol=1e-4)
        u_kernel = u_kernel(
            d=2,
            lengthscales=u_kernel_lengthscales,
            scale=u_kernel_scale,
            torchify=True,
            device=self.device,
        )
        kmat_u = torch.vstack(
            [
                torch.hstack(
                    [
                        u_kernel(
                            xticks[:, None, :],
                            xticks[None, :, :],
                            beta0=[0, 0],
                            beta1=[0, 0],
                        ),
                        u_kernel(
                            xticks[:, None, :],
                            xticks[None, :, :],
                            beta0=[0, 0],
                            beta1=[1, 0],
                        ),
                        u_kernel(
                            xticks[:, None, :],
                            xticks[None, :, :],
                            beta0=[0, 0],
                            beta1=[0, 1],
                        ),
                    ]
                ),
                torch.hstack(
                    [
                        u_kernel(
                            xticks[:, None, :],
                            xticks[None, :, :],
                            beta0=[1, 0],
                            beta1=[0, 0],
                        ),
                        u_kernel(
                            xticks[:, None, :],
                            xticks[None, :, :],
                            beta0=[1, 0],
                            beta1=[1, 0],
                        ),
                        u_kernel(
                            xticks[:, None, :],
                            xticks[None, :, :],
                            beta0=[1, 0],
                            beta1=[0, 1],
                        ),
                    ]
                ),
                torch.hstack(
                    [
                        u_kernel(
                            xticks[:, None, :],
                            xticks[None, :, :],
                            beta0=[0, 1],
                            beta1=[0, 0],
                        ),
                        u_kernel(
                            xticks[:, None, :],
                            xticks[None, :, :],
                            beta0=[0, 1],
                            beta1=[1, 0],
                        ),
                        u_kernel(
                            xticks[:, None, :],
                            xticks[None, :, :],
                            beta0=[0, 1],
                            beta1=[0, 1],
                        ),
                    ]
                ),
            ]
        ).detach() + u_kmat_noise * torch.eye(self.p2max_u, device=self.device)
        evals_u, evecs_u = torch.linalg.eigh(kmat_u)
        self.factor_u = evecs_u * torch.sqrt(evals_u)
        assert torch.allclose(self.factor_u @ self.factor_u.T, kmat_u, atol=1e-4)
        self.icdf_norm = torch.distributions.Normal(loc=0.0, scale=1.0).icdf
        self.x1meshes = [self.thin(l, x1mesh) for l in range(self.levels)]
        self.x2meshes = [self.thin(l, x2mesh) for l in range(self.levels)]
        self.lrs = [0.75] + [1] * max(0, self.levels - 1)
        self.num_newton_iters = np.array([30] + [5] * max(0, self.levels - 1))[
            : self.levels
        ]
        self.relaxations = np.array([1e-5] + [1e-8] * max(0, self.levels - 1))[
            : self.levels
        ]
        self.block_sizes = np.array([2**11, 2**10, 2**9])[: self.levels]

    def thin(self, level, x):
        import torch

        assert torch.get_default_dtype() == torch.float64
        if level == -1:
            level = self.levels - 1
        skip = 2 ** (self.levels - level - 1)
        return x[..., (skip - 1) :: skip, (skip - 1) :: skip]

    def transform_full(self, unifs):
        assert unifs.ndim >= 1 and unifs.shape[-1] == self.d
        shape = list(unifs.shape[:-1])
        unifs_flat = unifs.reshape((-1, self.d))
        normals = self.icdf_norm(unifs_flat)
        normals_u = normals[..., : self.p2max_u]
        normals_f = normals[..., self.p2max_u :]
        u = (self.factor_u @ normals_u.T).T.reshape(
            shape + [3, self.ps[-1], self.ps[-1]]
        )
        f = (self.factor_f @ normals_f.T).T.reshape(shape + [self.ps[-1], self.ps[-1]])
        return u, f

    def transform(self, level, unifs):
        u, f = self.transform_full(unifs)
        u_thin = self.thin(level, u)
        f_thin = self.thin(level, f)
        return u_thin, f_thin

    def draw_u_f(self, level=-1, shape=2):
        import torch

        if isinstance(shape, int):
            shape = [shape]
        unifs = torch.rand(list(shape) + [self.d], device=self.device)
        u_thin, f_thin = self.transform(level, unifs)
        return u_thin, f_thin

    def pde_solve(
        self,
        level,
        f,
        u,
        v0,
        lr=None,
        num_newton_iter=None,
        relaxation=None,
        verbose=False,
        verbose_indent=4,
    ):
        if lr is None:
            lr = self.lrs[level]
        if num_newton_iter is None:
            num_newton_iter = self.num_newton_iters[level]
        if relaxation is None:
            relaxation = self.relaxations[level]
        import torch

        assert f.ndim == 3 and u.ndim == 4 and v0.ndim == 3
        r = len(u)  # number of realizations
        n = self.ns[level] - 1
        n2 = n**2
        dx = self.dx[level]
        assert (
            f.shape == (r, n, n) and u.shape == (r, 3, n, n) and v0.shape == (r, n, n)
        )
        nrange = torch.arange(n, device=self.device)
        eyen2 = torch.eye(n2, device=self.device)
        # A_laplace.shape == (N,N)
        A1 = torch.zeros((n, n, n, n), device=self.device)
        for k in range(n):
            A1[nrange[1:], k, nrange[:-1], k] = 1
            A1[nrange, k, nrange, k] = -2
            A1[nrange[:-1], k, nrange[1:], k] = 1
        A2 = torch.zeros((n, n, n, n), device=self.device)
        for k in range(n):
            A2[k, nrange[1:], k, nrange[:-1]] = 1
            A2[k, nrange, k, nrange] = -2
            A2[k, nrange[:-1], k, nrange[1:]] = 1
        A_laplace = (A1 / dx**2 + A2 / dx**2).reshape((n2, n2))
        # C1.shape == C2.shape == (N,N)
        C1 = torch.zeros((n, n, n, n), device=self.device)
        C2 = torch.zeros((n, n, n, n), device=self.device)
        for k in range(n):
            C1[nrange[:-1], k, nrange[1:], k] = 1
            C1[nrange[1:], k, nrange[:-1], k] = -1
        for k in range(n):
            C2[k, nrange[:-1], k, nrange[1:]] = 1
            C2[k, nrange[1:], k, nrange[:-1]] = -1
        C1 = C1.reshape((n2, n2)) / (2 * dx)
        C2 = C2.reshape((n2, n2)) / (2 * dx)
        vs = torch.empty((r, num_newton_iter + 1, n2))
        resid = torch.empty((r, num_newton_iter + 1, n2))
        rmse_resid = torch.empty(r, num_newton_iter + 1)
        if verbose:
            _vstr = " " * verbose_indent + "%-15s| %-65s|" % (
                "iter of %-6d" % num_newton_iter,
                "RMSE residual",
            )
            print(_vstr)
            _vstr = " " * verbose_indent + "%-15s| %-13s%-13s%-13s%-13s%-13s|" % (
                " " * 15,
                "5%",
                "median",
                "mean",
                "95%",
                "finite %",
            )
            print(_vstr)
            _vstr = " " * verbose_indent + "-" * (len(_vstr) - verbose_indent)
            print(_vstr)
        # v.shape==(R,N)
        u = u.reshape((r, 3, n2))
        v = v0.reshape((r, n2))
        f = f.reshape((r, n2))

        def F(u, v, check=False):
            u0, u_x1, u_x2 = u[:, 0], u[:, 1], u[:, 2]
            C1v = torch.einsum("ik,rk->ri", C1, v)
            C2v = torch.einsum("ik,rk->ri", C2, v)
            Av = torch.einsum("ik,rk->ri", A_laplace, v)
            if check:
                _v = v.reshape((-1, n, n))
                _C1v = (_v[:, 2:, 1:9] - _v[:, :-2, 1:9]) / (2 * dx)
                assert torch.allclose(_C1v, C1v.reshape((-1, n, n))[:, 1:-1, 1:-1])
                _C2v = (_v[:, 1:9, 2:] - _v[:, 1:9, :-2]) / (2 * dx)
                assert torch.allclose(_C2v, C2v.reshape((-1, n, n))[:, 1:-1, 1:-1])
                _Av = (
                    _v[:, 2:, 1:9] - 2 * _v[:, 1:-1, 1:9] + _v[:, :-2, 1:9]
                ) / dx**2 + (
                    _v[:, 1:9, 2:] - 2 * _v[:, 1:9, 1:-1] + _v[:, 1:9, :-2]
                ) / dx**2
                assert torch.allclose(_Av, Av.reshape((-1, n, n))[:, 1:-1, 1:-1])
            y = (
                -torch.exp(u0) * (u_x1 * C1v + u_x2 * C2v + Av)
                + self.nonlinearity_factor * v**3
                - f
            )
            return y

        def partial_F(u, v):
            u0, u_x1, u_x2 = u[:, 0], u[:, 1], u[:, 2]
            y = -torch.exp(u0)[:, :, None] * (
                u_x1[:, :, None] * C1 + u_x2[:, :, None] * C2 + A_laplace
            ) + 3 * self.nonlinearity_factor * (v**2)[:, :, None] * torch.eye(
                n2, device=self.device
            )
            return y

        residual = F(u, v)
        for i in range(num_newton_iter + 1):
            vs[:, i, :] = v.cpu()
            resid[:, i, :] = residual.cpu()
            rmse_resid[:, i] = torch.sqrt(torch.mean(residual**2, 1)).cpu()
            if verbose and (i % verbose == 0 or i == num_newton_iter):
                _vstr = (
                    " " * verbose_indent
                    + "%-15d| %-13.2e%-13.2e%-13.2e%-13.2e%-13.1f|"
                    % (
                        i,
                        torch.nanquantile(rmse_resid[:, i], 0.05),
                        torch.nanquantile(rmse_resid[:, i], 0.5),
                        torch.nanmean(rmse_resid[:, i]),
                        torch.nanquantile(rmse_resid[:, i], 0.95),
                        100
                        * torch.mean(torch.isfinite(rmse_resid[:, i]).to(torch.float)),
                    )
                )
                print(_vstr)
            if i == num_newton_iter:
                break
            # dFdv.shape == (R,N,N)
            dFdv = partial_F(u, v)
            # Theta.shape==(R,N,N)
            Theta = torch.einsum("rki,rkj->rij", dFdv, dFdv)  # batch dFdv[r].T@dFdv[r]
            relaxation_to_try = relaxation
            while True:
                try:
                    # L.shape==Linv.shape==(R,N,N)
                    L = torch.linalg.cholesky(
                        Theta + relaxation_to_try * eyen2, upper=False
                    )
                    break
                except torch._C._LinAlgError as e:
                    # assert False, "level = %d"%level
                    expected_str = "linalg.cholesky: The factorization could not be completed because the input is not positive-definite"
                    if str(e)[: len(expected_str)] != expected_str:
                        raise
                    relaxation_to_try = 2 * relaxation_to_try
            Linv = torch.linalg.solve_triangular(
                L, torch.eye(L.size(-1), device=self.device), upper=False
            )
            # b.shape==delta.shape==v_new.shape==residual.shape==(R,N)
            b = torch.einsum("rik,ri->rk", dFdv, residual)
            delta = torch.einsum(
                "rik,ri->rk", Linv, torch.einsum("rik,rk->ri", Linv, b)
            )  # batch Linv[r].T@(Linv[r]@b[r])
            v_new = v - lr * delta
            residual_new = F(
                u, v_new
            )  # previously this was F(u,v) which I think is wrong
            v = v_new
            residual = residual_new
        vs = vs.reshape((r, num_newton_iter + 1, n, n))
        resid = resid.reshape((r, num_newton_iter + 1, n, n))
        data = {"vs": vs, "resid": resid, "rmse_resid": rmse_resid}
        return vs[:, -1, :, :], data

    def evaluate_from_u_f(self, level, u, f, pde_solve_kwargs={}):
        import torch

        assert f.shape[-2:] == u.shape[-2:]
        ogshape = u.shape[:-3]
        u = u.reshape([-1] + list(u.shape[-3:]))
        f = f.reshape([-1] + list(f.shape[-2:]))
        n = u.size(0)
        ys = []
        i = 0
        block_size = self.block_sizes[level]
        while i < n:
            i_next = min(i + block_size, n)
            f_i = f[i:i_next]
            u_i = u[i:i_next]
            v0_i = torch.ones(
                [u_i.size(0)] + [self.ns[level] - 1, self.ns[level] - 1],
                device=self.device,
            )
            y_i, data = self.pde_solve(
                level=level,
                f=f_i,
                u=u_i,
                v0=v0_i,
                **pde_solve_kwargs,
            )
            ys.append(y_i)
            i = i_next
        y = torch.cat(ys, 0)
        y = y.reshape(ogshape + y.shape[-2:])
        return y

    def __call__(self, level, samples=None, pde_solve_kwargs={}):
        import torch

        if samples is None:
            samples = torch.rand(self.d).to(self.device)
        npv = isinstance(samples, np.ndarray)
        if npv:
            samples = torch.from_numpy(samples).to(self.device)
        u, f = self.transform(level, samples)
        y = self.evaluate_from_u_f(level, u, f, pde_solve_kwargs)
        # h = self.ps[level]//4
        # qoi = y[...,h,h]**2
        qoi = y.amax((-2, -1))
        # qoi = y.std((-2,-1))
        if npv:
            qoi = qoi.cpu().numpy()
        return qoi

    def ml(self, level, samples=None, pde_solve_kwargs={}):
        import torch

        if samples is None:
            samples = torch.rand(self.d).to(self.device)
        npv = isinstance(samples, np.ndarray)
        if npv:
            samples = torch.from_numpy(samples).to(self.device)
        # h = self.ps[level]//4
        u_full, f_full = self.transform_full(samples)
        u_fine = self.thin(level, u_full)
        f_fine = self.thin(level, f_full)
        y_fine = self.evaluate_from_u_f(level, u_fine, f_fine, pde_solve_kwargs)
        # qoi_fine = y_fine[...,h,h]**2
        qoi_fine = y_fine.amax((-2, -1))
        # qoi_fine = y_fine.std((-2,-1))
        if level > 0:
            u_coarse = self.thin(level - 1, u_full)
            f_coarse = self.thin(level - 1, f_full)
            y_coarse = self.evaluate_from_u_f(
                level - 1, u_coarse, f_coarse, pde_solve_kwargs
            )
            # qoi_coarse = y_coarse[...,h,h]**2
            qoi_coarse = y_coarse.amax((-2, -1))
            # qoi_coarse = y_coarse.std((-2,-1))
            qoi = qoi_fine - qoi_coarse
        else:
            qoi = qoi_fine
        if npv:
            qoi = qoi.cpu().numpy()
        return qoi

    def plot_contour_grid(self, x, surface=True, contour_levels=250, figpath=None):
        import torch

        nrows = len(x)
        ncols = len(x[0])
        from matplotlib import pyplot

        fig = pyplot.figure()
        subplot_kw = {"projection": "3d"} if surface else {}
        fig, ax = pyplot.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6 * ncols, 6 * nrows),
            subplot_kw=subplot_kw,
        )
        ax = np.atleast_1d(ax).reshape((nrows, ncols))
        nlist = (self.ns - 1).tolist()
        for i in range(nrows):
            for j in range(ncols):
                xij = (
                    x[i][j]
                    if isinstance(x[i][j], np.ndarray)
                    else x[i][j].cpu().numpy()
                )
                this_n = xij.shape[-1]
                assert (
                    this_n in nlist
                ), "invalid x[%d][%d].shape[-1]=%d, must be in %s" % (
                    i,
                    j,
                    this_n,
                    str(nlist),
                )
                l = nlist.index(this_n)
                if surface:
                    ax[i, j].plot_surface(
                        self.x1meshes[l].cpu().numpy(),
                        self.x2meshes[l].cpu().numpy(),
                        xij,
                        cmap="gnuplot2",
                        rstride=1,
                        cstride=1,
                    )
                else:
                    ax[i, j].contourf(
                        self.x1meshes[l].cpu().numpy(),
                        self.x2meshes[l].cpu().numpy(),
                        xij,
                        cmap="gnuplot2",
                        levels=contour_levels,
                    )
        if figpath is not None:
            fig.savefig(figpath, bbox_inches="tight")
        return fig, ax


if __name__ == "__main__":
    import torch

    torch.set_default_dtype(torch.float64)
    df = DarcyFlow2d(device="cuda")
    # print(df.adjusted_costs)
    """ SOLVER TESTING """
    n = 100
    nplt = 8
    x = np.random.rand(n, df.d)
    us = [[None] * df.levels for i in range(n)]
    qs = [[None] * df.levels for i in range(n)]
    u, f = df.draw_u_f(level=-1, shape=n)
    us = [df.thin(l, u) for l in range(df.levels)]
    fs = [df.thin(l, f) for l in range(df.levels)]
    qs = [None] * df.levels
    for l in range(df.levels):
        u_l = us[l]
        print("u_l.shape = %s" % str(tuple(u_l.shape)))
        f_l = fs[l]
        print("f_l.shape = %s" % str(tuple(f_l.shape)))
        q_l = df.evaluate_from_u_f(
            level=l, u=u_l, f=f_l, pde_solve_kwargs={"verbose": True}
        )
        print("q_l.shape = %s" % str(tuple(q_l.shape)))
        qs[l] = q_l
        print()
    df.plot_contour_grid(
        [[fs[l][i] for l in range(df.levels)] for i in range(nplt)],
        figpath="darcy_f.png",
    )
    df.plot_contour_grid(
        [[us[l][i][0] for l in range(df.levels)] for i in range(nplt)],
        figpath="darcy_u.png",
    )
    df.plot_contour_grid(
        [[qs[l][i] for l in range(df.levels)] for i in range(nplt)],
        figpath="darcy_y.png",
    )
    """ MLQMC TESTING """
    qhat_prev = 0
    n = [2**19, 2**18, 2**13]
    # n = [2**15,2**14,2**9]
    print("MLQMC Test")
    for l in range(df.levels):
        x = qp.DigitalNetB2(df.d, seed=l)(n[l])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        q = df(level=l, samples=x)
        end.record()
        torch.cuda.synchronize()
        time_l = 0.001 * start.elapsed_time(end) / n[l]
        qhat_l = q.mean()
        yhat_l = qhat_l - qhat_prev
        print(
            "    n[%d] = %-10d Qhat[%d] = %-25.15f Yhat[%d] = %-25.15e time[%d] = %-.15e"
            % (l, n[l], l, qhat_l, l, yhat_l, l, time_l)
        )
        qhat_prev = qhat_l
