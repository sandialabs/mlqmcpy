import numpy as np 
import qmcpy as qp 

class DarcyFlow2d(object):
    def __init__(self, 
            levels = 3,
            n_coarsest = 4,
            nonlinearity_factor = 1.,
            f_kernel = qp.KernelMatern52,
            f_kernel_lengthscales = 0.1,
            f_kernel_scale = 1.,
            f_kmat_noise = 1e-5,
            f_seed = 1998,
            u_kernel = qp.KernelGaussian,
            u_kernel_lengthscales = 0.1, 
            u_kernel_scale = 1.,
            u_kmat_noise = 1e-5,
            device = "cpu",
            ):
        import torch
        assert torch.get_default_dtype()==torch.float64
        self.device = device
        rng_f = torch.Generator(device=self.device).manual_seed(f_seed)
        self.levels = levels 
        self.nonlinearity_factor = nonlinearity_factor
        assert n_coarsest>0 and np.log2(n_coarsest)%1==0
        self.ns = n_coarsest*2**np.arange(self.levels)
        x1 = torch.linspace(0,1,self.ns[-1]+1,device=self.device)[1:-1]
        x2 = torch.linspace(0,1,self.ns[-1]+1,device=self.device)[1:-1]
        self.dx = 1/self.ns
        self._d = (self.ns[-1]-1)**2
        self.d = 3*self._d
        x1mesh,x2mesh = torch.meshgrid(x1,x2,indexing="ij")
        xticks = torch.vstack([x1mesh.flatten(),x2mesh.flatten()]).T 
        f_kernel = f_kernel(d=2,lengthscales=f_kernel_lengthscales,scale=f_kernel_scale,torchify=True,device=self.device) 
        kmat_f = f_kernel(xticks[:,None,:],xticks[None,:,:]).detach()+f_kmat_noise*torch.eye(self._d,device=self.device)
        evals_f,evecs_f = torch.linalg.eigh(kmat_f)
        factor_f = evecs_f*torch.sqrt(evals_f)
        assert torch.allclose(factor_f@factor_f.T,kmat_f,atol=1e-4)
        f = (factor_f@torch.randn(self._d,generator=rng_f,device=self.device)).reshape((self.ns[-1]-1,self.ns[-1]-1))
        u_kernel = u_kernel(d=2,lengthscales=u_kernel_lengthscales,scale=u_kernel_scale,torchify=True,device=self.device)
        kmat_u = torch.vstack([
            torch.hstack([u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[0,0],beta1=[0,0]),u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[0,0],beta1=[1,0]),u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[0,0],beta1=[0,1])]),
            torch.hstack([u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[1,0],beta1=[0,0]),u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[1,0],beta1=[1,0]),u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[1,0],beta1=[0,1])]),
            torch.hstack([u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[0,1],beta1=[0,0]),u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[0,1],beta1=[1,0]),u_kernel(xticks[:,None,:],xticks[None,:,:],beta0=[0,1],beta1=[0,1])]),
        ]).detach()+u_kmat_noise*torch.eye(self.d,device=self.device)
        evals_u,evecs_u = torch.linalg.eigh(kmat_u)
        self.factor_u = evecs_u*torch.sqrt(evals_u)
        assert torch.allclose(self.factor_u@self.factor_u.T,kmat_u,atol=1e-4)
        self.icdf_norm = torch.distributions.Normal(loc=0.0, scale=1.0).icdf
        self.x1meshes = [self.thin(l,x1mesh) for l in range(self.levels)]
        self.x2meshes = [self.thin(l,x2mesh) for l in range(self.levels)]
        self.fs = [self.thin(l,f) for l in range(self.levels)]
    def thin(self, level, x):
        import torch
        assert torch.get_default_dtype()==torch.float64
        if level==-1:
            level = self.levels-1
        skip = 2**(self.levels-level-1)
        return x[...,(skip-1)::skip,(skip-1)::skip]
    def transform_u(self, level, unifs):
        import torch
        assert torch.get_default_dtype()==torch.float64
        assert unifs.ndim>=1 and unifs.shape[-1]==self.d
        shape = list(unifs.shape[:-1])
        unifs_flat = unifs.reshape((-1,self.d))
        normals = self.icdf_norm(unifs_flat)
        u = (self.factor_u@normals.T).T.reshape(shape+[3,self.ns[-1]-1,self.ns[-1]-1])
        u_thin = self.thin(level,u)
        return u_thin
    def draw_u(self, level=-1, shape=2):
        import torch
        assert torch.get_default_dtype()==torch.float64
        if isinstance(shape,int): shape=[shape]
        unifs = torch.rand(list(shape)+[self.d],device=self.device)
        u_thin = self.transform_u(level,unifs)
        return u_thin.detach().cpu().numpy()
    def pde_solve(
            self,
            level,
            f,
            u, 
            v0, 
            num_newton_iter = 5, 
            relaxation = 0, 
            lr = 1, 
            verbose = True,
            verbose_indent = 4,
            ):
        import torch
        assert torch.get_default_dtype()==torch.float64
        assert f.ndim==2 and u.ndim==4 and v0.ndim==3
        r = len(u) # number of realizations
        n = self.ns[level]-1
        n2 = n**2
        dx = self.dx[level]
        assert f.shape==(n,n) and u.shape==(r,3,n,n) and v0.shape==(r,n,n)
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
            y = -torch.exp(u0)*(u_x1*C1v+u_x2*C2v+Av)+self.nonlinearity_factor*v**3-f.flatten()
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
    def evaluate(self, level, samples=None, pde_solve_kwargs={}):
        import torch
        assert torch.get_default_dtype()==torch.float64
        if samples is None:
            samples = torch.rand(self.d).to(self.device)
        npv = isinstance(samples,np.ndarray)
        if npv: 
            samples = torch.from_numpy(samples).to(self.device)
        ogshape = samples.shape[:-1]
        u = self.transform_u(level,samples)
        u = u.reshape([-1]+list(u.shape[-3:]))
        y,data = self.pde_solve(
            level = level,
            f = self.fs[level],
            u = u, 
            v0 = torch.ones([u.size(0)]+[self.ns[level]-1,self.ns[level]-1],device=self.device),
            **pde_solve_kwargs,
            )
        y = y.reshape(ogshape+y.shape[-2:])
        if npv:
            y = y.cpu().numpy()
        return y
    def __call__(self, *args, **kwargs):
        import torch
        assert torch.get_default_dtype()==torch.float64
        y = self.evaluate(*args,**kwargs)
        if not isinstance(y,np.ndarray):
            y = y.cpu().numpy()
        qoi = y.max(axis=(-2,-1))
        return qoi
    def plot_contour_grid(self, x, contour_levels=250, figpath=None):
        import torch
        nrows = len(x)
        ncols = len(x[0])
        from matplotlib import pyplot
        fig,ax = pyplot.subplots(nrows=nrows,ncols=ncols,figsize=(6*ncols,6*nrows))
        ax = np.atleast_1d(ax).reshape((nrows,ncols))
        nlist =(self.ns-1).tolist()
        for i in range(nrows):
            for j in range(ncols):
                xij = x[i][j] if isinstance(x[i][j],np.ndarray) else x[i][j].cpu().numpy()
                this_n = xij.shape[-1]
                assert this_n in nlist, "invalid x[%d][%d].shape[-1]=%d, must be in %s"%(i,j,this_n,str(nlist))
                l = nlist.index(this_n)
                ax[i,j].contourf(self.x1meshes[l].cpu().numpy(),self.x2meshes[l].cpu().numpy(),xij,cmap="gnuplot2",levels=contour_levels)
        if figpath is not None:
            fig.savefig(figpath,bbox_inches="tight")
        return fig,ax

if __name__=="__main__":
    import torch 
    torch.set_default_dtype(torch.float64)
    df = DarcyFlow(device="cuda")
    n = 5
    x = np.random.rand(n,df.d)
    df.plot_contour_grid([df.fs],figpath="darcy_f.png")
    for l in range(df.levels):
        u_l = df.draw_u(l,shape=n)
        print("u_l.shape = %s"%str(tuple(u_l.shape)))
        df.plot_contour_grid(u_l,figpath="darcy_u_l%d.png"%l)
        y_l = df.evaluate(level=l,samples=x)
        print("y_l.shape = %s"%str(tuple(y_l.shape)))
        df.plot_contour_grid([y_l],figpath="darcy_y_l%d.png"%l)
        print()

