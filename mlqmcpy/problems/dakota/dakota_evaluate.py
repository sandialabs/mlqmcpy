import os
import subprocess
from tempfile import TemporaryDirectory

import numpy as np


def dakota_evaluate(dakota_in, **kwargs):

    # Replace keywords
    for key, val in kwargs.items():
        dakota_in = dakota_in.replace("{" + key + "}", val)

    with TemporaryDirectory() as tmp:

        # Write input file
        with open(os.path.join(tmp, "dakota.in"), "w") as io:
            io.write(dakota_in)

        # Execute Dakota
        try:
            command = "dakota dakota.in"
            subprocess.run(command, cwd=tmp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True, check=True)
        except subprocess.CalledProcessError:
            command = "source ~/.bash_profile; dakota dakota.in"
            subprocess.run(command, cwd=tmp, stdout=subprocess.DEVNULL, shell=True, check=True)

        # Extract results
        return np.genfromtxt(os.path.join(tmp, "output.txt"), comments="%")[:, -1]
