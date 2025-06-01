import time

import numpy as np
from matplotlib import pyplot as plt

from mlqmcpy.problems.dakota import steady_state_diffusion_1d

# rng = np.random.Generator(np.random.PCG64(7))
# solutions = steady_state_diffusion_1d(level=0,points=rng.uniform(0,1,(5,9)))
# print(solutions.shape)

num_points = 1000
num_dims = 9
num_levels = 8

points = np.random.rand(num_points, num_dims)
solutions = [None] * num_levels
times = [None] * num_levels
for level in range(num_levels):
    t0 = time.perf_counter()
    solutions[level] = steady_state_diffusion_1d(level, points)
    times[level] = time.perf_counter() - t0
    print(
        "level: %d\tsample mean: %.2e\ttime: %.1e\t time ratio: %.1f"
        % (
            level,
            solutions[level].mean(),
            times[level],
            -1 if level == 0 else times[level] / times[level - 1],
        )
    )

fig, axes = plt.subplots(1, num_levels, figsize=(5 * num_levels, 5))
for idx, ax in enumerate(axes.flatten()):
    ax.hist(solutions[idx], bins=16)
    ax.set_title(f"level = {idx}")
fig.tight_layout()
fig.savefig("plots/dakota_ssd1_scratch.hists.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(times)
ax.set_yscale("log")
fig.tight_layout()
fig.savefig("plots/dakota_ssd1_scratch.times.png")
