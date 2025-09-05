
# Studying the Generalization Behavior of Surrogate Models for Punch-Bending by Generating Plausible Counterfactuals

This repository contains scripts and the data to reproduced the results for the paper:

```
@article{mahajan2019preserving,
  title        = {Studying the Generalization Behavior of Surrogate Models for Punch-Bending by Generating Plausible Counterfactuals},
  author       = {Andreas Mazur, Henning Peters, André Artelt, Lukas Koller, Christoph Hartmann, Ansgar Trächtler and Barbara Hammer},
  booktitle    = {ICANN proceedings 2025}
  year         = {2025}
}
```

## How to use the code

The code is structured in the following way:
1. You need to run the training experiments first. This can be done by executing the script `run_experiments.py`.
   Depending on how you have specified your logging directory path, this will create a folder `logs` in which all the
   results are stored.
2. After the experiments are done, you can run the evaluation script `run_evaluation.py` to generate the plots. For this
   to work, you need to specify the path to the logging directory where the results are stored.

