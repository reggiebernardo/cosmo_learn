#!/bin/bash

# export PYTHONPATH=/path/to/your/cosmo_learn:$PYTHONPATH
export PYTHONPATH=/home/rbernardo/repos/cosmo_learn:$PYTHONPATH

python mocks.py
python hists_mcmc.py
python heat_mcmc.py
python gp_brr.py
python ann.py
python ga_fisher.py
