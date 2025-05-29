from tempfile import TemporaryDirectory
import subprocess
import os 
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
        # command = "source ~/.bash_profile; dakota %s/dakota.in"%tmp
        # command = "dakota -v"#%s/dakota.in"%tmp
        command = "dakota dakota.in"
        subprocess.run(command, cwd=tmp, stdout=subprocess.DEVNULL, shell=True, check=True)

        # Extract results
        return np.genfromtxt(os.path.join(tmp, "output.txt"), comments="%")