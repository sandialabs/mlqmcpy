import os

import numpy as np

from .dakota_evaluate import dakota_evaluate


def steady_state_diffusion_1d(
    level,
    points,
    template=os.path.join(os.path.dirname(__file__), "dakota.in.template"),
):
    # assert points.shape[-1]==9

    assert (0 <= points).all() and (points <= 1).all()

    points = 2 * points - 1
    _, num_terms = points.shape

    mesh_sizes = level * np.ones((points.shape[0], 1), dtype=int)
    points = np.hstack((points, mesh_sizes))

    # Read template file
    dakota_template_file = template
    with open(dakota_template_file, "r") as io:
        dakota_in = io.read()

    # Format list of points
    list_of_points = "\n"
    for point in points:
        list_of_points += " ".join(map(str, point)) + "\n"

    # Format variables
    variables = f"uniform_uncertain = {num_terms} \n\
	  lower_bounds      =  {num_terms}*-1. \n\
	  upper_bounds      =  {num_terms}* 1. \n\
    discrete_state_set \n\
	  integer = 1 \n\
	    set_values = 4 8 16 32 64 128 256 512 1024 2048 4096 \n\
	    descriptors = 'mesh_size'"

    # Evaluate dakota
    return dakota_evaluate(
        dakota_in,
        list_of_points=list_of_points,
        variables=variables,
        problem="'steady_state_diffusion_1d'",
    )[:, -1]
