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
    RidgeJSU,
    RidgePL,
    Genz,
    Sumxex,
    MC2,
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
import argparse

def main(problem_name, dimension, dataroot, trial_start, trial_end, true_solution, ref_approx_seed, device):
    if problem_name == "Analytic":
        problem = analytic
        assert dimension is None 
        dimension = 2
        num_levels = 4
        m_min = 2
        m_max = 15
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Borehole":
        problem = borehole
        assert dimension is None 
        dimension = 8
        num_levels = 2
        m_min = 2
        m_max = 13
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Elliptic PDE":
        problem = elliptic
        assert dimension is None 
        dimension = 8
        num_levels = 4
        m_min = 3
        m_max = 14
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Asian Option KL":
        problem = asian_option
        assert dimension is None 
        dimension = 16
        num_levels = 8
        m_min = 3
        m_max = 10
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Steady State Diffusion PDE":
        problem = steady_state_diffusion_1d
        assert dimension is None 
        dimension = 9
        num_levels = 5
        m_min = 2
        m_max = 9
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Asian Option":
        problem = MLFinancialOption(qmcpy_financial_option_args="ASIAN")
        assert dimension is None 
        dimension = problem.ds
        num_levels = problem.levels
        m_min = 4
        m_max = 11
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Lookback Option":
        problem = MLFinancialOption(qmcpy_financial_option_args="LOOKBACK")
        assert dimension is None 
        dimension = problem.ds
        num_levels = problem.levels
        m_min = 4
        m_max = 11
        cost_per_level = 2.**(np.arange(num_levels)-num_levels+1)
        initial_cost_prop_max_budget = 1/4
    elif problem_name == "Darcy Flow PDE 2D":
        assert device is not None, "Darcy Flow requires running on GPU"
        problem = DarcyFlow2d(device=device)
        assert dimension is None 
        dimension = problem.d
        num_levels = problem.levels
        m_min = 3
        m_max = 10
        cost_per_level = problem.adjusted_costs
        initial_cost_prop_max_budget = 1/4
    elif "Ridge" in problem_name:
        if problem_name == "Ridge Jump Equal Weights":
            problem = RidgeJump(weights="EQUAL")
        elif problem_name == "Ridge Kink Equal Weights":
            problem = RidgeKink(weights="EQUAL")
        elif problem_name == "Ridge Smooth Equal Weights":
            problem = RidgeSmooth(weights="EQUAL")
        elif problem_name == "Ridge Finance Equal Weights":
            problem = RidgeFinance(weights="EQUAL")
        elif problem_name == "Ridge JSU Equal Weights":
            problem = RidgeJSU(weights="EQUAL")
        elif problem_name == "Ridge PL Equal Weights":
            problem = RidgePL(weights="EQUAL")
        elif problem_name == "Ridge Jump Sparse Weights":
            problem = RidgeJump(weights="SPARSE")
        elif problem_name == "Ridge Kink Sparse Weights":
            problem = RidgeKink(weights="SPARSE")
        elif problem_name == "Ridge Smooth Sparse Weights":
            problem = RidgeSmooth(weights="SPARSE")
        elif problem_name == "Ridge Finance Sparse Weights":
            problem = RidgeFinance(weights="SPARSE")
        elif problem_name == "Ridge JSU Sparse Weights":
            problem = RidgeJSU(weights="SPARSE")
        elif problem_name == "Ridge PL Sparse Weights":
            problem = RidgePL(weights="SPARSE")
        else:
            raise Exception("invalid ridge function %s"%problem_name)
        assert isinstance(dimension,int)
        num_levels = 1
        m_min = 4
        m_max = 15
        cost_per_level = np.ones(1)
        initial_cost_prop_max_budget = 1
    elif problem_name=="Sumxex":
        problem = Sumxex()
        assert isinstance(dimension,int)
        num_levels = 1
        m_min = 4
        m_max = 15
        cost_per_level = np.ones(1)
        initial_cost_prop_max_budget = 1
    elif problem_name=="MC2":
        problem = MC2()
        assert isinstance(dimension,int)
        num_levels = 1
        m_min = 4
        m_max = 15
        cost_per_level = np.ones(1)
        initial_cost_prop_max_budget = 1
    elif "Genz" in problem_name:
        assert isinstance(dimension,int)
        if problem_name == "Genz Oscillatory 1":
            problem = Genz(dimension,kind_func="OSCILLATORY",kind_coeff=1)
        elif problem_name == "Genz Oscillatory 2":
            problem = Genz(dimension,kind_func="OSCILLATORY",kind_coeff=2)
        elif problem_name == "Genz Oscillatory 3":
            problem = Genz(dimension,kind_func="OSCILLATORY",kind_coeff=3)
        elif problem_name == "Genz Corner-Peak 1":
            problem = Genz(dimension,kind_func="CORNER-PEAK",kind_coeff=1)
        elif problem_name == "Genz Corner-Peak 2":
            problem = Genz(dimension,kind_func="CORNER-PEAK",kind_coeff=2)
        elif problem_name == "Genz Corner-Peak 3":
            problem = Genz(dimension,kind_func="CORNER-PEAK",kind_coeff=3)
        else:
            raise Exception("invalid Genz function %s"%problem_name)
        num_levels = 1
        m_min = 4
        m_max = 15
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
    fname = dataroot+"trials.%d.%d.log"%(trial_start,trial_end)
    print("\t%s"%fname)
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
        ("MC",mp.GreedyMLMCIterator,{},None),
        ## RQMC
        ##  LATTICE
        # (r"RQMC Lat $R=2$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":2},"BAKER"),
        # (r"RQMC Lat $R=4$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":4},"BAKER"),
        (r"RQMC Lat      ",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":8},"BAKER"),
        # (r"RQMC Lat $R=16$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":16},"BAKER"),
        # (r"RQMC Lat $R=32$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":32},"BAKER"),
        # (r"RQMC Lat $R=64$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"replications":64},"BAKER"),
        ##  DNET 
        # (r"RQMC Net $R=2$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":2},None),
        # (r"RQMC Net $R=4$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":4},None),
        (r"RQMC Net      ",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":8},None),
        # (r"RQMC Net $R=16$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":16},None),
        # (r"RQMC Net $R=32$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":32},None),
        # (r"RQMC Net $R=64$",mp.GreedyMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"replications":64},None),
        ## GPQMC 
        ##   FAST
        ##       LATTICE 
        (r"GPQMC Lat           ",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},"BAKER"),
        # (r"GPQMC Lat $\alpha=2$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},"BAKER"),
        # (r"GPQMC Lat $\alpha=3$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},"BAKER"),
        # (r"GPQMC Lat $\alpha=4$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},"BAKER"),
        # (r"GPQMC Lat Combined",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.Lattice,"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":kwargs_kernel_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps,"kernel_class":qp.KernelShiftInvarCombined},"BAKER"),
        ##       DNET
        # (r"GPQMC Net Adaptive  ",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"kwargs_discrete_distrib_construct":{"alpha":1,**kwargs_discrete_distrib_construct},"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":kwargs_kernel_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps,"kernel_class":qp.KernelDigShiftInvarAdaptiveAlpha},None),
        # (r"GPQMC Net $\alpha=1$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"kwargs_discrete_distrib_construct":{"alpha":1,**kwargs_discrete_distrib_construct},"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"GPQMC Net $\alpha=2$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"kwargs_discrete_distrib_construct":{"alpha":1,**kwargs_discrete_distrib_construct},"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"GPQMC Net $\alpha=3$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"kwargs_discrete_distrib_construct":{"alpha":1,**kwargs_discrete_distrib_construct},"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        # (r"GPQMC Net $\alpha=4$",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"kwargs_discrete_distrib_construct":{"alpha":1,**kwargs_discrete_distrib_construct},"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps},None),
        (r"GPQMC Net           ",mp.GreedyGaussianProcessMLQMCIterator,{"discrete_distribution_type":qp.DigitalNetB2,"kwargs_discrete_distrib_construct":{"alpha":1,**kwargs_discrete_distrib_construct},"fast":True,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":kwargs_kernel_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_igps,"kernel_class":qp.KernelDigShiftInvarCombined},None),
        ## MTGPF
        ##   FAST 
        ##       LATTICE
        # (r"MTGPF Lat $\alpha=1$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPF Lat $\alpha=2$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPF Lat $\alpha=3$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPF Lat $\alpha=4$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # ##       DNET
        # (r"MTGPF  Net $\alpha=1$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPF  Net $\alpha=2$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPF  Net $\alpha=3$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPF  Net $\alpha=4$",mp.MultiTaskGaussianProcessMLQMCIteratorFunction,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # MTGPD
        #   FAST 
        #       LATTICE
        # (r"MTGPD Lat $\alpha=1$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPD Lat $\alpha=2$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPD Lat $\alpha=3$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPD Lat $\alpha=4$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.Lattice,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        #       DNET
        # (r"MTGPD Net $\alpha=1$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":1,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPD Net $\alpha=2$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":2,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPD Net $\alpha=3$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":3,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
        # (r"MTGPD Net $\alpha=4$",mp.MultiTaskGaussianProcessMLQMCIteratorDifference,{"discrete_distribution_type":qp.DigitalNetB2,"fast":True,"budget_scheme":mtgp_budget_scheme,"kwargs_fastgp_construct":kwargs_fastgp_construct,"kwargs_kernel_construct":{"alpha":4,**kwargs_kernel_construct},"kwargs_kernel_mt_construct":kwargs_kernel_mt_construct,"kwargs_fastgp_fit":kwargs_fastgp_fit,"refit_gps":refit_mtgps},None),
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
    parser = argparse.ArgumentParser(description="Budgeted Comparison Experiment")
    parser.add_argument(
        "-p",
        "--problem",
        type = str,
        default = "Ridge Jump Equal Weights",
        help = "problem name"
    )
    parser.add_argument(
        "-d",
        "--dimension",
        type = int,
        default = None,
        help = "problem dimension if applicable"
    )
    parser.add_argument(
        "-f",
        "--force",
        action='store_true',
        help = "Flag to force experiment to overwrite existing folder"
    )
    parser.add_argument(
        "--outdir",
        type = str,
        default = "",
        help = "output directory"
    )
    parser.add_argument(
        "--tag",
        type = str,
        default = "",
        help = "tag"
    )
    parser.add_argument(
        "-t",
        "--trials",
        type = int,
        default = 100,
        help = "trials"
    )
    parser.add_argument(
        "--parallel",
        type = int,
        default = 1,
        help = "parallel processes, 1 means serial execution"
    )
    parser.add_argument(
        "--devices",
        nargs = "+",
        type = int,
        default = -1,
        help = "devices with (-1) the CPU and a non-negative int the Cuda GPU index"
    )
    parser.add_argument(
        "--nrefapprox",
        type = int,
        default = 2**19,
        help = "seed for the reference approximation if required"
    )
    parser.add_argument(
        "--refapproxseed",
        type = int,
        default = 7,
        help = "seed for the reference approximation if required"
    )
    args = parser.parse_args()
    assert args.parallel>0
    if isinstance(args.devices,int):
        args.devices = [args.devices]*args.parallel
    devices = [("cpu" if args.devices[i]==-1 else "cuda:%d") for i in range(args.parallel)]
    assert len(devices)>=args.parallel
    # directory setup
    dataroot = os.path.dirname(os.path.abspath(__file__))+"/budgeted_comparison_data/"+args.outdir+"%s%s/"%(args.problem.replace(" ","_"),args.tag)
    if os.path.exists(dataroot) and (not args.force):
        print("\n\nexperiment %s exists, ending program\n\n"%dataroot)
        sys.exit(0)
    if os.path.exists(dataroot):
        shutil.rmtree(dataroot)
    os.makedirs(dataroot)
    # approximate true solution 
    if args.parallel==1:
        main(args.problem,args.dimension,dataroot,0,args.nrefapprox,None,args.refapproxseed,devices[0])
    else:
        bs = int(np.ceil(args.nrefapprox/args.parallel))
        n_blocks = [(i*bs,min(args.nrefapprox,(i+1)*bs)) for i in range(args.parallel)]
        processes = [torch.multiprocessing.Process(target=main,args=(args.problem,args.dimension,dataroot,n_min,n_max,None,args.refapproxseed,devices[i])) for i,(n_min,n_max) in enumerate(n_blocks)]
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
    true_solution = true_solution/args.nrefapprox
    # run ML(Q)MC simulations
    if args.parallel==1:
        main(args.problem,args.dimension,dataroot,0,args.trials,true_solution,None,devices[0])
    else:
        bs = int(np.ceil(args.trials/args.parallel))
        trial_blocks = [(i*bs,min(args.trials,(i+1)*bs)) for i in range(args.parallel)]
        processes = [torch.multiprocessing.Process(target=main,args=(args.problem,args.dimension,dataroot,trial_start,trial_end,true_solution,None,devices[i])) for i,(trial_start,trial_end) in enumerate(trial_blocks)]
        for p in processes: p.start()
        for p in processes: p.join()
    print()
    costs = []
    means = []
    std_errors = []
    true_errors = []
    samples_per_level = []
    for file in os.listdir(dataroot):
        fparts = file.split('.')
        if fparts[0]!="data" or fparts[-1]!="npy": continue
        trial_start,trial_end = int(fparts[1]),int(fparts[2])
        data_f = np.load(dataroot+"data.%d.%d.npy"%(trial_start,trial_end),allow_pickle=True)[()]
        problem_name = data_f["problem_name"]
        dimension = data_f["dimension"]
        num_levels = data_f["num_levels"]
        true_solution = data_f["true_solution"]
        initial_cost_prop_max_budget = data_f["initial_cost_prop_max_budget"]
        max_budgets = data_f["max_budgets"]
        names = data_f["names"]
        costs.append(data_f["costs"])
        means.append(data_f["means"])
        std_errors.append(data_f["std_errors"])
        true_errors.append(data_f["true_errors"])
        kwargs_fastgp_fit = data_f["kwargs_fastgp_fit"]
        refit_igps = data_f["refit_gps"]
        samples_per_level.append(data_f["samples_per_level"])
    costs = np.concatenate(costs,-1)
    means = np.concatenate(means,-1)
    std_errors = np.concatenate(std_errors,-1)
    true_errors = np.concatenate(true_errors,-1)
    samples_per_level = np.concatenate(samples_per_level,-2)
    trials = costs.shape[-1]
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
        "refit_gps": refit_igps,
        "trials": costs.shape[-1]
    }
    np.save(dataroot+"data.npy",data)
    for file in os.listdir(dataroot):
        if file=="data.npy" or file[-3:]=="log": continue 
        os.remove(dataroot+file)

