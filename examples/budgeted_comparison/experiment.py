import mlqmcpy as mp
from mlqmcpy.problems import analytic, elliptic, asian_option, borehole, steady_state_diffusion_1d

import numpy as np
import scipy.stats
import qmcpy as qp
import time
import os
import shutil
import gc
import sys


def main(problem, dimension, num_levels, dataroot, trial_start, trial_end):
    assert trial_end>trial_start
    trials = trial_end-trial_start
    # parameters 
    # max_budgets = 2**np.arange(8,15)
    max_budgets = 2**np.arange(9,12)
    initial_cost_prop_max_budget = 1/4
    kwargs_discrete_distrib_construct = {}
    kwargs_kernel_construct = {"requires_grad_scale":True,"requires_grad_lengthscales":True}
    kwargs_kernel_mt_construct = {"rank_factor":num_levels}
    kwargs_fastgp_construct = {"requires_grad_noise":False}
    kwargs_fastgp_fit = {"loss_metric":"MLL","stop_crit_improvement_threshold":1e0,"verbose":0}
    refit_igps = True
    refit_mtgps = True
    initial_sampling_alloc = "PROP" # ["PROP","EQUAL"]
    mtgp_budget_scheme = "GREEDY" # ["GREEDY","FULL"]
    n_ref = 2**19
    verbose = max(1,trials//10)
    zip_name_IteratorClass_kwargs = [
        ## MLMC
        # ("MLMC    IID",mp.GreedyMLMCIterator,{},None),
        ## R-MLQMC
        ##  LATTICE
        (r"R-MLQMC Lattice $R=8$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":8},None),
        (r"R-MLQMC Lattice $R=16$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":16},None),
        # (r"R-MLQMC Lattice $R=32$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":32},None),
        # (r"R-MLQMC Lattice $R=8$ Baker",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":8},"BAKER"),
        # (r"R-MLQMC Lattice $R=16$ Baker",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":16},"BAKER"),
        # (r"R-MLQMC Lattice $R=32$ Baker",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":32},"BAKER"),
        ##  DNET 
        (r"R-MLQMC DNet $R=8$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":8},None),
        (r"R-MLQMC DNet $R=16$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":16},None),
        # (r"R-MLQMC DNet $R=32$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":32},None),
        ## IGP 
        ##   FAST
        ##       LATTICE 
        # (r"IGP    Lattice  Fast  SI  $\alpha=1$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"IGP    Lattice  Fast  SI  $\alpha=2$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"IGP    Lattice  Fast  SI  $\alpha=3$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"IGP    Lattice  Fast  SI  $\alpha=4$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"IGP    Lattice  Fast  SI  $\alpha=1$ BAKER",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},"BAKER"),
        # (r"IGP    Lattice  Fast  SI  $\alpha=2$ BAKER",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},"BAKER"),
        # (r"IGP    Lattice  Fast  SI  $\alpha=3$ BAKER",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},"BAKER"),
        # (r"IGP    Lattice  Fast  SI  $\alpha=4$ BAKER",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},"BAKER"),
        ##       DNET
        # (r"IGP    DNet     Fast  DSI $\alpha=1$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"IGP    DNet     Fast  DSI $\alpha=2$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"IGP    DNet     Fast  DSI $\alpha=3$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"IGP    DNet     Fast  DSI $\alpha=4$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        ## MTGPF
        ##   FAST 
        ##       LATTICE
        # (r"MTGPF  Lattice  Fast  SI  $\alpha=1$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPF  Lattice  Fast  SI  $\alpha=2$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPF  Lattice  Fast  SI  $\alpha=3$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPF  Lattice  Fast  SI  $\alpha=4$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # ##       DNET
        # (r"MTGPF  DNet     Fast  DSI $\alpha=1$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPF  DNet     Fast  DSI $\alpha=2$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPF  DNet     Fast  DSI $\alpha=3$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPF  DNet     Fast  DSI $\alpha=4$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # MTGPD
        #   FAST 
        #       LATTICE
        # (r"MTGPD  Lattice  Fast  SI  $\alpha=1$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPD  Lattice  Fast  SI  $\alpha=2$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPD  Lattice  Fast  SI  $\alpha=3$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPD  Lattice  Fast  SI  $\alpha=4$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        #       DNET
        # (r"MTGPD  DNet     Fast  DSI $\alpha=1$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPD  DNet     Fast  DSI $\alpha=2$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPD  DNet     Fast  DSI $\alpha=3$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPD  DNet     Fast  DSI $\alpha=4$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
    ]
    # experiment
    t0 = time.perf_counter()
    names = [name for (name,IteratorClass,kwargs,tf_type) in zip_name_IteratorClass_kwargs]
    if hasattr(problem,"exact") and hasattr(problem.exact,"Q") and hasattr(problem.exact.Q,"mean"):
        true_solution = problem.exact.Q.mean(level=num_levels-1)
    else:
        true_solution = problem(num_levels-1,qp.DigitalNetB2(dimension)(n_ref)).mean()
    cost_per_level = 2.**(np.arange(num_levels)-num_levels+1) # Rescale so that finest level has unit cost
    print("max_budgets: %s"%max_budgets)
    print("cost_per_level: %s"%str(cost_per_level))
    print("true_solution = %.5e"%true_solution)
    print()
    costs = np.empty((len(zip_name_IteratorClass_kwargs),len(max_budgets),trials))
    means = np.empty((len(zip_name_IteratorClass_kwargs),len(max_budgets),trials))
    std_errors = np.empty((len(zip_name_IteratorClass_kwargs),len(max_budgets),trials))
    true_errors = np.empty((len(zip_name_IteratorClass_kwargs),len(max_budgets),trials))
    samples_per_level = np.empty((len(zip_name_IteratorClass_kwargs),len(max_budgets),trials,num_levels),dtype=int)
    for j in range(len(max_budgets)):
        max_budget = max_budgets[j]
        initial_budget = int(initial_cost_prop_max_budget*max_budget)
        print("budget = %d, initial budget = %d"%(max_budget,initial_budget))
        if initial_sampling_alloc.upper()=="PROP":
            initial_sample_size_og = initial_budget/num_levels/cost_per_level
        elif initial_sampling_alloc.upper()=="EQUAL":
            initial_sample_size_og = initial_budget/num_levels*np.ones(num_levels)
        else:
            assert False,"initial_sample_size_og should be 'PROP' or 'EQUAL'"
        for i,(name,IteratorClass,kwargs,tf_type) in enumerate(zip_name_IteratorClass_kwargs):
            if IteratorClass==mp.GreedyMLQMCIterator:
                initial_sample_size = initial_sample_size_og/kwargs["replications"]
                initial_sample_size = 2**np.ceil(np.log2(initial_sample_size)).astype(int)
                initial_cost = kwargs["replications"]*(initial_sample_size*cost_per_level).sum()
            else:
                initial_sample_size = 2**np.ceil(np.log2(initial_sample_size_og)).astype(int)
                initial_cost = (initial_sample_size*cost_per_level).sum()
            print("\t%s, \t initial_sample_size = %s, \tinitial cost = %.1f"%(name,str(initial_sample_size.tolist()),initial_cost))
            for t in range(trials):
                iterator = IteratorClass(
                    dimension,
                    initial_sample_size = initial_sample_size,
                    cost_per_level = cost_per_level,
                    max_budget = max_budget,
                    **kwargs)
                problem_l_or_ml = problem if isinstance(iterator,mp.MultiTaskGaussianProcessMLQMCIteratorFunction) else problem.ml
                problem_tf = problem_l_or_ml
                if tf_type is None:
                    problem_tf = problem_l_or_ml
                elif tf_type.upper()=="BAKER":
                    problem_tf = lambda level,x: problem_l_or_ml(level,1-2*np.abs(x-1/2))
                else:
                    raise Exception("tf_type %s not available"%str(tf_type))
                for iter, new_samples in enumerate(iterator):
                    new_results = {level: problem_tf(level,samples) for level,samples in new_samples.items()}
                    iterator.update(new_results)
                costs[i,j,t] = iterator.cost
                means[i,j,t] = iterator.mean
                std_errors[i,j,t] = iterator.standard_error
                true_errors[i,j,t] = np.abs(true_solution-means[i,j,t])
                samples_per_level[i,j,t] = iterator.total_samples_per_level.copy()
                if verbose and t%verbose==0:
                    print("\t\ttrial: %-6d iteration: %-6d cost: %-10d mean: %-15.3e std error: %-15.3e true error: %-15.3e time %-15d sample sizes %s"%\
                    (t,iter,costs[i,j,t],means[i,j,t],std_errors[i,j,t],true_errors[i,j,t],int(np.ceil(time.perf_counter()-t0)),str(samples_per_level[i,j,t].tolist())))
                #import psutil; process = psutil.Process(os.getpid()); print(f"Total program memory: {process.memory_info().rss / (1024 * 1024):.2f} MB")
                # del iterator
                # import gc; gc.collect()
        print()
    data = {
        "problem_name": problem.__name__,
        "dimension": dimension,
        "num_levels": num_levels,
        "max_budgets": max_budgets,
        "initial_cost_prop_max_budget": initial_cost_prop_max_budget,
        "names": names,
        "costs": costs,
        "means": means,
        "std_errors": std_errors,
        "samples_per_level": samples_per_level,
        "true_errors": true_errors,
        "kwargs_fastgp_fit": kwargs_fastgp_fit,
        "refit_gps":refit_igps,
        "n_ref":n_ref,
    }
    np.save(dataroot+"data.%d.%d.npy"%(trial_start,trial_end),data)

if __name__=="__main__":
    force_experiment = True
    parallel = False
    problem_dim_levels = (analytic,2,4)
    #problem_dim_levels = (borehole,8,2)
    #problem_dim_levels = (elliptic,8,4)
    #problem_dim_levels = (asian_option,16,8)
    #problem_dim_levels = (steady_state_diffusion_1d,9,5)
    problem,dimension,num_levels = problem_dim_levels
    dataroot = os.path.dirname(os.path.abspath(__file__))+"/budgeted_comparison_data/comp.%s.d%d.levels%d.TMP/"%(problem.__name__,dimension,num_levels)
    # directory setup
    if os.path.exists(dataroot) and (not force_experiment):
        print("experiment exists, ending program")
        sys.exit(0)
    if os.path.exists(dataroot):
        shutil.rmtree(dataroot)
    os.makedirs(dataroot)
    trials = 5
    trial_blocks = [(i,i+1) for i in range(trials)]
    if parallel:
        assert False 
    else:
        for trial_start,trial_end in trial_blocks:
            # run experiments 
            main(problem, dimension, num_levels, dataroot, trial_start, trial_end)
