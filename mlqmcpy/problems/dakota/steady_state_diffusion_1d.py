from .dakota_evaluate import dakota_evaluate
import numpy as np 
import os

def steady_state_diffusion_1d(level, points, template="dakota.in.template"):
    """
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> solutions = steady_state_diffusion_1d(level=0,points=rng.uniform(0,1,(5,9)))
    >>> solutions
    array([0.11922965, 0.12787293, 0.12074919, 0.11519112, 0.1311341 ])
    """
    assert points.shape[-1]==9
    assert (0<=points).all() and (points<=1).all()
    points = 2*points-1
    
    # Read template file
    dakota_template_file = os.path.dirname(os.path.realpath(__file__))+"/"+template

    with open(dakota_template_file, "r") as io:
        dakota_in = io.read()

    # Format list of points
    list_of_points = "\n"
    for point in points:
        list_of_points += " ".join(map(str, point)) + " 0 \n"

    # Format variables
    _, num_terms = points.shape
    variables = f"uniform_uncertain = {num_terms} \n\
	  lower_bounds      =  {num_terms}*-1. \n\
	  upper_bounds      =  {num_terms}* 1. \n\
    discrete_state_set \n\
	  integer = 1 \n\
	    set_values = {2 * 2**level} \n\
	    descriptors = 'mesh_size'"
    
    # Evaluate dakota
    return dakota_evaluate(dakota_in, list_of_points=list_of_points, variables=variables, problem="'steady_state_diffusion_1d'")[:, -1]
