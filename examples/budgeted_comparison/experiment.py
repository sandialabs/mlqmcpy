import mlqmcpy as mp
from mlqmcpy.problems import (
    analytic,
    elliptic,
    asian_option,
    borehole,
    steady_state_diffusion_1d,
    MLFinancialOption,
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

def main(problem_name, problem, dimension, num_levels, m_min, m_max, true_solution, dataroot, trial_start, trial_end):
    assert trial_end>trial_start
    trials = trial_end-trial_start
    fname = dataroot+"log.%d.%d.log"%(trial_start,trial_end)
    print("%s"%fname)
    file = open(fname,"w")
    # parameters 
    max_budgets = 2**np.arange(m_min,m_max)
    initial_cost_prop_max_budget = 1/4
    kwargs_discrete_distrib_construct = {}
    kwargs_kernel_construct = {"requires_grad_scale":True,"requires_grad_lengthscales":True}
    kwargs_kernel_mt_construct = {"rank_factor":num_levels}
    kwargs_fastgp_construct = {"requires_grad_noise":False}
    kwargs_fastgp_fit = {"loss_metric":"MLL","stop_crit_improvement_threshold":1e-1,"verbose":0}
    refit_igps = True
    refit_mtgps = True
    initial_sampling_alloc = "PROP" # ["PROP","EQUAL"]
    mtgp_budget_scheme = "GREEDY" # ["GREEDY","FULL"]
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
    cost_per_level = 2.**(np.arange(num_levels)-num_levels+1) # Rescale so that finest level has unit cost
    file.write("max_budgets: %s\n"%max_budgets)
    file.write("cost_per_level: %s\n"%np.array_repr(cost_per_level).replace('\n', ''))
    file.write("true_solution = %.5e\n"%true_solution)
    file.write("\n"+("*"*100)+"\n\n")
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
        file.write("max_budget = %d, trying sample_size_alloc = %s\n\n"%(max_budget,np.array_repr(sample_size_alloc.astype(int)).replace('\n', '')))
        for i,(name,IteratorClass,kwargs,tf_type) in enumerate(zip_name_IteratorClass_kwargs):
            if IteratorClass==mp.GreedyMLQMCIterator:
                initial_sampling_alloc_r = sample_size_alloc/kwargs["replications"]
                if (initial_sampling_alloc_r<1).any():
                    continue
                icpmb2 = max(initial_cost_prop_max_budget,1/initial_sampling_alloc_r.min())
                initial_sample_size = icpmb2*initial_sampling_alloc_r
                initial_sample_size = 2**np.ceil(np.log2(initial_sample_size)).astype(int)
                initial_cost = kwargs["replications"]*(initial_sample_size*cost_per_level).sum()
            else:
                if (sample_size_alloc<2).any():
                    continue
                icpmb2 = max(initial_cost_prop_max_budget,2/sample_size_alloc.min())
                initial_sample_size = icpmb2*sample_size_alloc
                initial_sample_size = 2**np.ceil(np.log2(initial_sample_size)).astype(int)
                initial_cost = (initial_sample_size*cost_per_level).sum()
            assert initial_cost<=max_budget
            file.write("\t%s, \t initial_sample_size = %s, \tinitial cost = %.1f\n"%(name,np.array_repr(initial_sample_size).replace('\n', ''),initial_cost))
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
                costs[i,j,t] = iterator.cost
                means[i,j,t] = iterator.mean
                std_errors[i,j,t] = iterator.standard_error
                true_errors[i,j,t] = np.abs(true_solution-means[i,j,t])
                samples_per_level[i,j,t] = iterator.total_samples_per_level.copy()
                if verbose and t%verbose==0:
                    file.write("\t\ttrial: %-6d iteration: %-6d cost: %-10d mean: %-15.3e std error: %-15.3e true error: %-15.3e time %-15d sample sizes %s\n"%\
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
    force_experiment = False
    tag = "NEW"
    trials = 25
    parallel = 1

    if False:
        problem_name = "Analytic"
        problem = analytic
        dimension = 2
        num_levels = 4
        m_min = 2
        m_max = 15
    elif False:
        problem_name = "Borehole"
        problem = borehole
        dimension = 8
        num_levels = 2
        m_min = 2
        m_max = 13
        n_ref_approx = 2**19
    elif False:
        problem_name = "Elliptic PDE"
        problem = elliptic
        dimension = 8
        num_levels = 4
        m_min = 2
        m_max = 14
        n_ref_approx = 2**19
    elif False:
        problem_name = "Asian Option KL"
        problem = asian_option
        dimension = 16
        num_levels = 8
        m_min = 3
        m_max = 10
        n_ref_approx = 2**19
    elif False:
        problem_name = "Steady State Diffusion PDE"
        problem = steady_state_diffusion_1d
        dimension = 9
        num_levels = 5
        m_min = 2
        m_max = 9
        problem_dim_levels_ms = (steady_state_diffusion_1d,9,5,2,9)
        n_ref_approx = 2**18
    elif False:
        problem_name = "Asian Option"
        problem = MLFinancialOption(qmcpy_financial_option_args="ASIAN")
        dimension = problem.ds
        num_levels = problem.levels
        m_min = 4
        m_max = 10
    elif True:
        problem_name = "Lookback Option"
        problem = MLFinancialOption(qmcpy_financial_option_args="LOOKBACK")
        dimension = problem.ds
        num_levels = problem.levels
        m_min = 4
        m_max = 10
        n_ref_approx = 2**19
    else:
        raise Exception("please set one of the problem cases to true")
    print()
    dataroot = os.path.dirname(os.path.abspath(__file__))+"/budgeted_comparison_data/comp.%s.%s/"%(problem_name,tag)
    # directory setup
    if os.path.exists(dataroot) and (not force_experiment):
        print("experiment %s exists, ending program"%dataroot)
        sys.exit(0)
    if os.path.exists(dataroot):
        shutil.rmtree(dataroot)
    os.makedirs(dataroot)
    if hasattr(problem,"exact") and hasattr(problem.exact,"Q") and hasattr(problem.exact.Q,"mean"):
        true_solution = problem.exact.Q.mean(level=num_levels-1)
    else:
        d = dimension if isinstance(dimension,int) else dimension[-1]
        true_solution = problem(num_levels-1,qp.DigitalNetB2(d)(n_ref_approx)).mean()
    assert parallel>0
    bs = int(np.ceil(trials/parallel))
    trial_blocks = [(i*bs,min(trials,(i+1)*bs)) for i in range(parallel)]
    if parallel==1:
        for trial_start,trial_end in trial_blocks:
            # run experiments 
            main(problem_name, problem, dimension, num_levels, m_min, m_max, true_solution, dataroot, trial_start, trial_end)
    else:
        print("%d CPUs available, using parallel = %d CPUs"%(os.cpu_count(),parallel))
        processes = []
        for trial_start,trial_end in trial_blocks:
            process = multiprocessing.Process(target=main, args=(problem_name, problem, dimension, num_levels, m_min, m_max, true_solution, dataroot, trial_start, trial_end))
            processes.append(process)
        for process in processes:
            process.start()
        for process in processes:
            process.join()
    print()
        
