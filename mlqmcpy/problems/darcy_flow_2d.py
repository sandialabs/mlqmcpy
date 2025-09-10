import numpy as np 
import qmcpy as qp 

class DarcyFlow2d(object):
    def __init__(self, 
            levels = 4,
            n_coarsest = 4,
            nonlinearity_factor = 1,
            f_kernel = qp.KernelMatern12,
            f_kernel_lengthscales = 0.1,
            f_kernel_scale = 1.,
            f_kmat_noise = 1e-5,
            u_kernel = qp.KernelGaussian,
            u_kernel_lengthscales = 0.1, 
            u_kernel_scale = 1.,
            u_kmat_noise = 1e-5,
            device = "cpu",
            ):
        import torch
        assert torch.get_default_dtype()==torch.float64
        self.device = device
        self.levels = levels 
        self.nonlinearity_factor = nonlinearity_factor
        assert n_coarsest>0 and np.log2(n_coarsest)%1==0
        self.ns = n_coarsest*2**np.arange(self.levels)
        self.dx = 1/self.ns
        self.ps = self.ns-1
        self.pmax = self.ps[-1]
        self.p2s = self.ps**2
        self.p2max = self.p2s[-1]
        self.p2max_u = 3*self.p2max
        self.d = self.p2max+self.p2max_u
        x1 = torch.linspace(0,1,self.ns[-1]+1,device=self.device)[1:-1]
        x2 = torch.linspace(0,1,self.ns[-1]+1,device=self.device)[1:-1]
        x1mesh,x2mesh = torch.meshgrid(x1,x2,indexing="ij")
        xticks = torch.vstack([x1mesh.flatten(),x2mesh.flatten()]).T 
        f_kernel = f_kernel(d=2,lengthscales=f_kernel_lengthscales,scale=f_kernel_scale,torchify=True,device=self.device) 
        kmat_f = f_kernel(xticks[:,None,:],xticks[None,:,:]).detach()+f_kmat_noise*torch.eye(self.p2max,device=self.device)
        evals_f,evecs_f = torch.linalg.eigh(kmat_f)
        self.factor_f = evecs_f*torch.sqrt(evals_f)
        assert torch.allclose(self.factor_f@self.factor_f.T,kmat_f,atol=1e-4)
        u_kernel = u_kernel(d=2,lengthscales=u_kernel_lengthscales,scale=u_kernel_scale,torchify=True,device=self.device)
        kmat_u = torch.vstack([
            torch.hstack([u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[0,0],beta1=[0,0]),u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[0,0],beta1=[1,0]),u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[0,0],beta1=[0,1])]),
            torch.hstack([u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[1,0],beta1=[0,0]),u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[1,0],beta1=[1,0]),u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[1,0],beta1=[0,1])]),
            torch.hstack([u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[0,1],beta1=[0,0]),u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[0,1],beta1=[1,0]),u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[0,1],beta1=[0,1])]),
        ]).detach()+u_kmat_noise*torch.eye(self.p2max_u,device=self.device)
        evals_u,evecs_u = torch.linalg.eigh(kmat_u)
        self.factor_u = evecs_u*torch.sqrt(evals_u)
        assert torch.allclose(self.factor_u@self.factor_u.T,kmat_u,atol=1e-4)
        self.icdf_norm = torch.distributions.Normal(loc=0.0, scale=1.0).icdf
        self.x1meshes = [self.thin(l,x1mesh) for l in range(self.levels)]
        self.x2meshes = [self.thin(l,x2mesh) for l in range(self.levels)]
        self.lrs = [.9,.75]+[1]*max(0,self.levels-2)
        self.num_newton_iters = np.array([100,75]+[5]*max(0,self.levels-2))[:self.levels]
        self.relaxations = np.array([0,1e-5]+[0]*max(0,self.levels-2))[:self.levels]
        self.raw_costs = self.num_newton_iters*self.p2s**3
        self.adjusted_costs = self.raw_costs/self.raw_costs[-1]
    def thin(self, level, x):
        import torch
        assert torch.get_default_dtype()==torch.float64
        if level==-1:
            level = self.levels-1
        skip = 2**(self.levels-level-1)
        return x[...,(skip-1)::skip,(skip-1)::skip]
    def transform(self, level, unifs):
        import torch
        assert torch.get_default_dtype()==torch.float64
        assert unifs.ndim>=1 and unifs.shape[-1]==self.d
        shape = list(unifs.shape[:-1])
        unifs_flat = unifs.reshape((-1,self.d))
        normals = self.icdf_norm(unifs_flat)
        normals_u = normals[...,:self.p2max_u]
        normals_f = normals[...,self.p2max_u:]
        u = (self.factor_u@normals_u.T).T.reshape(shape+[3,self.ps[-1],self.ps[-1]])
        f = (self.factor_f@normals_f.T).T.reshape(shape+[self.ps[-1],self.ps[-1]])
        u_thin = self.thin(level,u)
        f_thin = self.thin(level,f)
        return u_thin,f_thin
    def draw_u_f(self, level=-1, shape=2):
        import torch
        assert torch.get_default_dtype()==torch.float64
        if isinstance(shape,int): shape=[shape]
        unifs = torch.rand(list(shape)+[self.d],device=self.device)
        u_thin,f_thin = self.transform(level,unifs)
        return u_thin,f_thin
    def pde_solve(
            self,
            level,
            f,
            u, 
            v0, 
            lr = None, 
            num_newton_iter = None, 
            relaxation = None, 
            verbose = False,
            verbose_indent = 4,
            ):
        if lr is None:
            lr = self.lrs[level]
        if num_newton_iter is None:
            num_newton_iter = self.num_newton_iters[level]
        if relaxation is None:
            relaxation = self.relaxations[level]
        import torch
        assert torch.get_default_dtype()==torch.float64
        assert f.ndim==3 and u.ndim==4 and v0.ndim==3
        r = len(u) # number of realizations
        n = self.ns[level]-1
        n2 = n**2
        dx = self.dx[level]
        assert f.shape==(r,n,n) and u.shape==(r,3,n,n) and v0.shape==(r,n,n)
        nrange = torch.arange(n,device=self.device)
        n2range = torch.arange(n2,device=self.device)
        # A_laplace.shape == (N,N)
        A1 = torch.zeros((n,n,n,n),device=self.device)
        for k in range(n):
            A1[nrange[1:],k,nrange[:-1],k] = 1
            A1[nrange,k,nrange,k] = -2
            A1[nrange[:-1],k,nrange[1:],k] = 1
        A2 = torch.zeros((n,n,n,n),device=self.device)
        for k in range(n):
            A2[k,nrange[1:],k,nrange[:-1]] = 1
            A2[k,nrange,k,nrange] = -2
            A2[k,nrange[:-1],k,nrange[1:]] = 1
        A_laplace = (A1/dx**2+A2/dx**2).reshape((n2,n2))
        # C1.shape == C2.shape == (N,N)
        C1 = torch.zeros((n,n,n,n),device=self.device) 
        C2 = torch.zeros((n,n,n,n),device=self.device) 
        for k in range(n):
            C1[nrange[:-1],k,nrange[1:],k] = 1
            C1[nrange[1:],k,nrange[:-1],k] = -1
        for k in range(n):
            C2[k,nrange[:-1],k,nrange[1:]] = 1
            C2[k,nrange[1:],k,nrange[:-1]] = -1
        C1 = C1.reshape((n2,n2))/(2*dx)
        C2 = C2.reshape((n2,n2))/(2*dx)
        vs = torch.empty((r,num_newton_iter+1,n2))
        resid = torch.empty((r,num_newton_iter+1,n2))
        rmse_resid = torch.empty(r,num_newton_iter+1)
        if verbose:
            _vstr = " "*verbose_indent+"%-15s| %-65s|"%("iter of %-6d"%num_newton_iter,"RMSE residual")
            print(_vstr)
            _vstr = " "*verbose_indent+"%-15s| %-10s| %-13s%-13s%-13s%-13s%-13s|"%(" "*15," "*10,"5%","median","mean","95%","finite %")
            print(_vstr)
            _vstr = " "*verbose_indent+"-"*(len(_vstr)-verbose_indent)
            print(_vstr)
        # v.shape==(R,N)
        u = u.reshape((r,3,n2))
        v = v0.reshape((r,n2))
        f = f.reshape((r,n2))
        def F(u, v, check=False):
            u0,u_x1,u_x2 = u[:,0],u[:,1],u[:,2]
            C1v = torch.einsum("ik,rk->ri",C1,v)
            C2v = torch.einsum("ik,rk->ri",C2,v)
            Av = torch.einsum("ik,rk->ri",A_laplace,v)
            if check:
                _v = v.reshape((-1,n,n))
                _C1v = (_v[:,2:,1:9]-_v[:,:-2,1:9])/(2*dx) 
                assert torch.allclose(_C1v,C1v.reshape((-1,n,n))[:,1:-1,1:-1])
                _C2v = (_v[:,1:9,2:]-_v[:,1:9,:-2])/(2*dx) 
                assert torch.allclose(_C2v,C2v.reshape((-1,n,n))[:,1:-1,1:-1])
                _Av = (_v[:,2:,1:9]-2*_v[:,1:-1,1:9]+_v[:,:-2,1:9])/dx**2+(_v[:,1:9,2:]-2*_v[:,1:9,1:-1]+_v[:,1:9,:-2])/dx**2
                assert torch.allclose(_Av,Av.reshape((-1,n,n))[:,1:-1,1:-1])
            y = -torch.exp(u0)*(u_x1*C1v+u_x2*C2v+Av)+self.nonlinearity_factor*v**3-f
            return y
        def partial_F(u, v):
            u0,u_x1,u_x2 = u[:,0],u[:,1],u[:,2]
            y = -torch.exp(u0)[:,:,None]*(u_x1[:,:,None]*C1+u_x2[:,:,None]*C2+A_laplace)+3*self.nonlinearity_factor*(v**2)[:,:,None]*torch.eye(n2,device=self.device)
            return y
        residual = F(u,v)
        for i in range(num_newton_iter+1):
            vs[:,i,:] = v.cpu()
            resid[:,i,:] = residual.cpu()
            rmse_resid[:,i] = torch.sqrt(torch.mean(residual**2,1)).cpu()
            if verbose and (i%verbose==0 or i==num_newton_iter):
                _vstr = " "*verbose_indent+"%-15d| %-13.2e%-13.2e%-13.2e%-13.2e%-13.1f|"%(i,
                        torch.nanquantile(rmse_resid[:,i],.05),torch.nanquantile(rmse_resid[:,i],.5),torch.nanmean(rmse_resid[:,i]),torch.nanquantile(rmse_resid[:,i],.95),100*torch.mean(torch.isfinite(rmse_resid[:,i]).to(torch.float)))
                print(_vstr)
            if i==num_newton_iter: break
            # dFdv.shape == (R,N,N)
            dFdv = partial_F(u,v)
            # Theta.shape==(R,N,N)
            Theta = torch.einsum("rki,rkj->rij",dFdv,dFdv) # batch dFdv[r].T@dFdv[r]
            Theta[:,n2range,n2range] = Theta[:,n2range,n2range]+relaxation # add to diagonals
            # L.shape==Linv.shape==(R,N,N)
            L = torch.linalg.cholesky(Theta,upper=False)
            Linv = torch.linalg.solve_triangular(L,torch.eye(L.size(-1),device=self.device),upper=False)
            # b.shape==delta.shape==v_new.shape==residual.shape==(R,N)
            b = torch.einsum("rik,ri->rk",dFdv,residual)
            delta = torch.einsum("rik,ri->rk",Linv,torch.einsum("rik,rk->ri",Linv,b)) # batch Linv[r].T@(Linv[r]@b[r])
            v_new = v-lr*delta
            residual_new = F(u,v_new) # previously this was F(u,v) which I think is wrong
            v = v_new 
            residual = residual_new
        vs = vs.reshape((r,num_newton_iter+1,n,n))
        resid = resid.reshape((r,num_newton_iter+1,n,n))
        data = {"vs":vs,"resid":resid,"rmse_resid":rmse_resid}
        return vs[:,-1,:,:],data
    def evaluate_from_u_f(self, level, u, f, pde_solve_kwargs={}):
        import torch
        assert torch.get_default_dtype()==torch.float64
        assert f.shape[-2:]==u.shape[-2:]
        ogshape = u.shape[:-3]
        u = u.reshape([-1]+list(u.shape[-3:]))
        f = f.reshape([-1]+list(f.shape[-2:]))
        y,data = self.pde_solve(
            level = level,
            f = f,
            u = u, 
            v0 = torch.ones([u.size(0)]+[self.ns[level]-1,self.ns[level]-1],device=self.device),
            **pde_solve_kwargs,
            )
        y = y.reshape(ogshape+y.shape[-2:])
        return y
    def __call__(self, level, samples=None, pde_solve_kwargs={}):
        import torch
        assert torch.get_default_dtype()==torch.float64
        if samples is None:
            samples = torch.rand(self.d).to(self.device)
        npv = isinstance(samples,np.ndarray)
        if npv: 
            samples = torch.from_numpy(samples).to(self.device)
        u,f = self.transform(level,samples)
        y = self.evaluate_from_u_f(level,u,f,pde_solve_kwargs)
        qoi = y.amax((-2,-1))
        if npv:
            qoi = qoi.cpu().numpy()
        return qoi
    def plot_contour_grid(self, x, surface=True, contour_levels=250, figpath=None):
        import torch
        nrows = len(x)
        ncols = len(x[0])
        from matplotlib import pyplot
        fig = pyplot.figure()
        subplot_kw = {'projection':'3d'} if surface else {}
        fig,ax = pyplot.subplots(nrows=nrows,ncols=ncols,figsize=(6*ncols,6*nrows),subplot_kw=subplot_kw)
        ax = np.atleast_1d(ax).reshape((nrows,ncols))
        nlist =(self.ns-1).tolist()
        for i in range(nrows):
            for j in range(ncols):
                xij = x[i][j] if isinstance(x[i][j],np.ndarray) else x[i][j].cpu().numpy()
                this_n = xij.shape[-1]
                assert this_n in nlist, "invalid x[%d][%d].shape[-1]=%d, must be in %s"%(i,j,this_n,str(nlist))
                l = nlist.index(this_n)
                if surface:
                    ax[i,j].plot_surface(self.x1meshes[l].cpu().numpy(),self.x2meshes[l].cpu().numpy(),xij,cmap="gnuplot2",rstride=1,cstride=1)
                else:
                    ax[i,j].contourf(self.x1meshes[l].cpu().numpy(),self.x2meshes[l].cpu().numpy(),xij,cmap="gnuplot2",levels=contour_levels)
        if figpath is not None:
            fig.savefig(figpath,bbox_inches="tight")
        return fig,ax

if __name__=="__main__":
    import torch 
    torch.set_default_dtype(torch.float64)
    df = DarcyFlow2d(device="cuda")
    print(df.raw_costs)
    print(df.adjusted_costs)
    """ SOLVER TESTING """ 
    n = 1000
    nplt = 5
    x = np.random.rand(n,df.d)
    us = [[None]*df.levels for i in range(n)]
    ys = [[None]*df.levels for i in range(n)]
    u,f = df.draw_u_f(level=-1,shape=n)
    us = [df.thin(l,u) for l in range(df.levels)]
    fs = [df.thin(l,f) for l in range(df.levels)]
    ys = [None]*df.levels
    for l in range(df.levels):
        u_l = us[l]
        print("u_l.shape = %s"%str(tuple(u_l.shape)))
        f_l = fs[l]
        print("f_l.shape = %s"%str(tuple(f_l.shape)))
        y_l = df.evaluate_from_u_f(level=l,u=u_l,f=f_l,pde_solve_kwargs={"verbose":True})
        print("y_l.shape = %s"%str(tuple(y_l.shape)))
        ys[l] = y_l
        print()
    df.plot_contour_grid([[fs[l][i] for l in range(df.levels)] for i in range(nplt)],figpath="darcy_f.png")
    df.plot_contour_grid([[us[l][i][0] for l in range(df.levels)] for i in range(nplt)],figpath="darcy_u.png")
    df.plot_contour_grid([[ys[l][i] for l in range(df.levels)] for i in range(nplt)],figpath="darcy_y.png")
    """ MLQMC TESTING """ 
    qhat_prev = 0
    x = qp.DigitalNetB2(df.d,seed=7)(2**10)
    print("MLQMC Test with x.shape = %s"%str(x.shape))
    for l in range(df.levels):
        q = df(level=l,samples=x)
        qhat_l = q.mean()
        yhat_l = qhat_l-qhat_prev
        print("    Qhat[l] = %-10.3f Yhat[l] = %.2e"%(qhat_l,yhat_l))
        qhat_prev = qhat_l


