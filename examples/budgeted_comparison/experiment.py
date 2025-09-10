import torch 
import mlqmcpy as mp
from mlqmcpy.problems import (
    analytic,
    elliptic,
    asian_option,
    borehole,
    steady_state_diffusion_1d,
    MLFinancialOption,
    DarcyFlow2d,
    RidgeJump,
    RidgeFinance,
    RidgeKink,
    RidgeSmooth,
)

import numpy as np
import scipy.stats
import qmcpy as qp
import time
import os
import shutil
import gc
import sys
import multiprocessing

def main(problem_name, dataroot, trial_start, trial_end, true_solution, ref_approx_seed, device):
    if problem_name == "Analytic":
        problem = analytic
        dimension = 2
        num_levels = 4
        m_min = 2
        m_max = 15
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Borehole":
        problem = borehole
        dimension = 8
        num_levels = 2
        m_min = 2
        m_max = 13
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Elliptic PDE":
        problem = elliptic
        dimension = 8
        num_levels = 4
        m_min = 2
        m_max = 14
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Asian Option KL":
        problem = asian_option
        dimension = 16
        num_levels = 8
        m_min = 3
        m_max = 10
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Steady State Diffusion PDE":
        problem = steady_state_diffusion_1d
        dimension = 9
        num_levels = 5
        m_min = 2
        m_max = 9
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Asian Option":
        problem = MLFinancialOption(qmcpy_financial_option_args="ASIAN")
        dimension = problem.ds
        num_levels = problem.levels
        m_min = 4
        m_max = 10
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Lookback Option":
        problem = MLFinancialOption(qmcpy_financial_option_args="LOOKBACK")
        dimension = problem.ds
        num_levels = problem.levels
        m_min = 4
        m_max = 10
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Darcy Flow PDE 2D":
        assert device is not None, "Darcy Flow requires running on GPU"
        problem = DarcyFlow2d(device=device)
        dimension = problem.d
        num_levels = problem.levels
        m_min = 3
        m_max = 10
        cost_per_level = problem.adjusted_costs
        initial_cost_prop_max_budget = 1/4
    elif "Ridge" in problem_name:
        if problem_name == "Ridge Jump":
            problem = RidgeJump()
        elif problem_name == "Ridge Kink":
            problem = RidgeKink()
        elif problem_name == "Ridge Smooth":
            problem = RidgeSmooth()
        elif problem_name == "Ridge Finance":
            problem = RidgeFinance()
        else:
            raise Exception("invalid ridge problem %s"%problem_name)
        dimension = 2
        num_levels = 1
        m_min = 4
        m_max = 13
        cost_per_level = np.ones(1)
        initial_cost_prop_max_budget = 1
    else:
        raise Exception("invalid problem_name = %s"%problem_name)
    if true_solution is None:
        if hasattr(problem,"exact") and hasattr(problem.exact,"Q") and hasattr(problem.exact.Q,"mean"):
            ymean = problem.exact.Q.mean(level=num_levels-1)
        else:
            d = dimension if isinstance(dimension,int) else dimension[-1]
            dnb2 = qp.DigitalNetB2(d,order="GRAY",seed=ref_approx_seed)
            x = dnb2(n_min=trial_start,n_max=trial_end)
            y = problem(num_levels-1,x)
            ymean = y.mean()
        data_ref_approx = {"ymean":ymean}
        np.save(dataroot+"ymean.%d.%d.npy"%(trial_start,trial_end),data_ref_approx)
        return 
    torch.set_default_dtype(torch.float64)
    assert trial_end>trial_start
    trials = trial_end-trial_start
    fname = dataroot+"log.%d.%d.log"%(trial_start,trial_end)
    print("%s"%fname)
    file = open(fname,"w")
    # parameters 
    initial_sampling_alloc = "PROP" # ["PROP","EQUAL"]
    mtgp_budget_scheme = "GREEDY" # ["GREEDY","FULL"]
    max_budgets = 2**np.arange(m_min,m_max)
    kwargs_discrete_distrib_construct = {}
    kwargs_kernel_construct = {
        "device": device,
        "requires_grad_scale": True,
        "requires_grad_lengthscales": True
    }
    kwargs_kernel_mt_construct = {"rank_factor":num_levels}
    kwargs_fastgp_construct = {"requires_grad_noise":False}
    kwargs_fastgp_fit = {
        "loss_metric": "MLL",
        "stop_crit_improvement_threshold": 1e-1,
        "verbose": 0,
        # "lr": 1e0,
    }
    refit_igps = True
    refit_mtgps = True
    verbose = max(1,trials//10)
    zip_name_IteratorClass_kwargs = [
        ## MLMC
        ("MLMC    IID",mp.GreedyMLMCIterator,{},None),
        ## R-MLQMC
        ##  LATTICE
        # (r"R-MLQMC Lattice $R=2$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":2},None),
        # (r"R-MLQMC Lattice $R=4$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":4},None),
        # (r"R-MLQMC Lattice $R=8$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":8},None),
        # (r"R-MLQMC Lattice $R=16$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":16},None),
        # (r"R-MLQMC Lattice $R=2$ Baker",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":2},"BAKER"),
        # (r"R-MLQMC Lattice $R=4$ Baker",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":4},"BAKER"),
        # (r"R-MLQMC Lattice $R=8$ Baker",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":8},"BAKER"),
        # (r"R-MLQMC Lattice $R=16$ Baker",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":16},"BAKER"),
        ##  DNET 
        (r"R-MLQMC DNet $R=2$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":2},None),
        (r"R-MLQMC DNet $R=4$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":4},None),
        (r"R-MLQMC DNet $R=8$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":8},None),
        (r"R-MLQMC DNet $R=16$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":16},None),
        (r"R-MLQMC DNet $R=32$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":32},None),
        (r"R-MLQMC DNet $R=64$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":64},None),
        ## IGP 
        ##   FAST
        ##       LATTICE 
        # (r"IGP    Lattice  Fast  SI  $\alpha=1$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"IGP    Lattice  Fast  SI  $\alpha=2$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"IGP    Lattice  Fast  SI  $\alpha=3$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"IGP    Lattice  Fast  SI  $\alpha=4$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"IGP    Lattice  Fast  SI  $\alpha=1$ Baker",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},"BAKER"),
        # (r"IGP    Lattice  Fast  SI  $\alpha=2$ Baker",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},"BAKER"),
        # (r"IGP    Lattice  Fast  SI  $\alpha=3$ Baker",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},"BAKER"),
        # (r"IGP    Lattice  Fast  SI  $\alpha=4$ Baker",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},"BAKER"),
        ##       DNET
        (r"IGP    DNet     Fast  DSI Adaptive  ",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"kwargs_discrete_distrib_construct":{"alpha":1,**kwargs_discrete_distrib_construct},"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps,"kernel_class":qp.KernelDigShiftInvarAdaptiveAlpha},None),
        (r"IGP    DNet     Fast  DSI $\alpha=1$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"kwargs_discrete_distrib_construct":{"alpha":1,**kwargs_discrete_distrib_construct},"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        (r"IGP    DNet     Fast  DSI $\alpha=2$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"kwargs_discrete_distrib_construct":{"alpha":1,**kwargs_discrete_distrib_construct},"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        (r"IGP    DNet     Fast  DSI $\alpha=3$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"kwargs_discrete_distrib_construct":{"alpha":1,**kwargs_discrete_distrib_construct},"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        (r"IGP    DNet     Fast  DSI $\alpha=4$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"kwargs_discrete_distrib_construct":{"alpha":1,**kwargs_discrete_distrib_construct},"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
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
    file.write("true_solution = %.5e\n"%true_solution)
    file.write("cost_per_level: %s\n"%np.array_repr(cost_per_level).replace('\n', ''))
    file.write("max_budgets: %s\n"%max_budgets)
    file.write("\n")
    costs = np.nan*np.ones((len(zip_name_IteratorClass_kwargs),len(max_budgets),trials))
    means = np.nan*np.ones((len(zip_name_IteratorClass_kwargs),len(max_budgets),trials))
    std_errors = np.nan*np.ones((len(zip_name_IteratorClass_kwargs),len(max_budgets),trials))
    true_errors = np.nan*np.ones((len(zip_name_IteratorClass_kwargs),len(max_budgets),trials))
    samples_per_level = np.nan*np.ones((len(zip_name_IteratorClass_kwargs),len(max_budgets),trials,num_levels),dtype=int)
    for j in range(len(max_budgets)):
        max_budget = max_budgets[j]
        if initial_sampling_alloc.upper()=="PROP":
            sample_size_alloc = max_budget/num_levels/cost_per_level
        elif initial_sampling_alloc.upper()=="EQUAL":
            sample_size_alloc = max_budget/num_levels*np.ones(num_levels)
        else:
            assert False,"initial_sampling_alloc should be 'PROP' or 'EQUAL'"
        if (sample_size_alloc<1).any():
            raise Exception("try increasing minimum budget: using max_budget = %d with %s allocation gives invalid sample sizes  %s"%(max_budget,initial_sampling_alloc,sample_size_alloc)) 
        file.write("\tmax_budget = %d, trying sample_size_alloc = %s\n\n"%(max_budget,np.array_repr(sample_size_alloc.astype(int)).replace('\n', '')))
        for i,(name,IteratorClass,kwargs,tf_type) in enumerate(zip_name_IteratorClass_kwargs):
            if IteratorClass==mp.GreedyMLQMCIterator:
                initial_sampling_alloc_r = sample_size_alloc/kwargs["replications"]
                if (initial_sampling_alloc_r<1).any():
                    continue
                icpmb2 = max(initial_cost_prop_max_budget,1/initial_sampling_alloc_r.min())
                initial_sample_size = icpmb2*initial_sampling_alloc_r
                initial_sample_size = 2**np.floor(np.log2(initial_sample_size)).astype(int)
                initial_cost = kwargs["replications"]*(initial_sample_size*cost_per_level).sum()
            else:
                if (sample_size_alloc<2).any():
                    continue
                icpmb2 = max(initial_cost_prop_max_budget,2/sample_size_alloc.min())
                initial_sample_size = icpmb2*sample_size_alloc
                initial_sample_size = 2**np.floor(np.log2(initial_sample_size)).astype(int)
                initial_cost = (initial_sample_size*cost_per_level).sum()
            assert initial_cost<=max_budget
            file.write("\t\t%s, \t initial_sample_size = %s, \tinitial cost = %.1f\n"%(name,np.array_repr(initial_sample_size).replace('\n', ''),initial_cost))
            file.flush()
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
                if iterator.standard_error==0:
                    pass
                costs[i,j,t] = iterator.cost
                means[i,j,t] = iterator.mean
                std_errors[i,j,t] = iterator.standard_error
                true_errors[i,j,t] = np.abs(true_solution-means[i,j,t])
                samples_per_level[i,j,t] = iterator.total_samples_per_level.copy()
                if verbose and t%verbose==0:
                    file.write("\t\t\ttrial: %-6d iteration: %-6d cost: %-10d mean: %-15.3e std error: %-15.3e true error: %-15.3e time %-15d sample sizes %s\n"%\
                    (t,iter,costs[i,j,t],means[i,j,t],std_errors[i,j,t],true_errors[i,j,t],int(np.ceil(time.perf_counter()-t0)),np.array_repr(samples_per_level[i,j,t].astype(int)).replace('\n', '')))
                    file.flush()
                #import psutil; process = psutil.Process(os.getpid()); file.write(f"Total program memory: {process.memory_info().rss / (1024 * 1024):.2f} MB\n")
                # del iterator
                # import gc; gc.collect()
        file.write("\n")
        file.flush()
    file.close()
    data = {
        "problem_name": problem_name,
        "dimension": dimension,
        "num_levels": num_levels,
        "true_solution": true_solution,
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
    }
    np.save(dataroot+"data.%d.%d.npy"%(trial_start,trial_end),data)

if __name__=="__main__":
    torch.set_default_dtype(torch.float64)
    force_experiment = True
    folder = "ridge_SL/d2/"
    tag = "NEW"
    trials = 100
    parallel = 10
    # devices = ["cuda:3","cuda:4"]
    devices = "cpu"
    ref_approx_seed = 7
    problem_name,n_ref_approx = (
        # "Analytic",None
        # "Borehole",2**19
        # "Elliptic PDE",2**19
        # "Asian Option KL",2**19
        # "Steady State Diffusion PDE",2**18
        # "Asian Option",None
        # "Lookback Option",2**19
        # "Darcy Flow PDE 2D",2**15
        "Ridge Jump", 2**20
        # "Ridge Kink", 2**20
        # "Ridge Smooth", 2**20
        # "Ridge Finance", 2**20
    )
    print()
    if isinstance(devices,str): devices = [devices]*parallel
    assert len(devices)>=parallel
    # directory setup
    dataroot = os.path.dirname(os.path.abspath(__file__))+"/budgeted_comparison_data/"+folder+"comp.%s.%s/"%(problem_name,tag)
    if os.path.exists(dataroot) and (not force_experiment):
        print("experiment %s exists, ending program"%dataroot)
        sys.exit(0)
    if os.path.exists(dataroot):
        shutil.rmtree(dataroot)
    os.makedirs(dataroot)
    assert parallel>0
    # approximate true solution 
    if parallel==1:
        main(problem_name, dataroot, 0, n_ref_approx, None, ref_approx_seed, devices[0])
    else:
        bs = int(np.ceil(n_ref_approx/parallel))
        n_blocks = [(i*bs,min(n_ref_approx,(i+1)*bs)) for i in range(parallel)]
        processes = [torch.multiprocessing.Process(target=main,args=(problem_name,dataroot,n_min,n_max,None,ref_approx_seed, devices[i])) for i,(n_min,n_max) in enumerate(n_blocks)]
        for p in processes: p.start()
        for p in processes: p.join()
    true_solution = 0
    for file in os.listdir(dataroot):
        fparts = file.split('.')
        if fparts[0]!="ymean" or fparts[-1]!="npy": continue
        n_min,n_max = int(fparts[1]),int(fparts[2])
        data = np.load(dataroot+"ymean.%d.%d.npy"%(n_min,n_max),allow_pickle=True)[()]
        ymean = data["ymean"]
        true_solution += ymean*(n_max-n_min)
    true_solution = true_solution/n_ref_approx
    # run ML(Q)MC simulations
    if parallel==1:
        main(problem_name, dataroot, 0, trials, true_solution, None, devices[0])
    else:
        bs = int(np.ceil(trials/parallel))
        trial_blocks = [(i*bs,min(trials,(i+1)*bs)) for i in range(parallel)]
        processes = [torch.multiprocessing.Process(target=main,args=(problem_name,dataroot,trial_start,trial_end,true_solution,None,devices[i])) for i,(trial_start,trial_end) in enumerate(trial_blocks)]
        for p in processes: p.start()
        for p in processes: p.join()
    print()
        
