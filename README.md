# cs181-p4
Practical 4: Reinforcement Learning for CS 181.  
Team Members: Vincent Li, Harry Fu, Nathan Zhao

## Instructions  
1. Run agent with `python Q_learn.py`.  
2. Change baseline hyperparameter and feature configurations in `Q_learn.py`.  
  a. Change hyperparameters in lines 13-18.  
  b. Change features in `tuple_generate` function in lines 46-96.  
3. Change number of epochs in line 205 (3rd parameter, default is 100).
4. IMPORTANT: Change name of `.npy` file for each trial (to avoid overwriting data) in line 208.
5. Plot and save histogram while calculating metrics with `python plot_hist.py NAME_OF_NPY_FILE`. Ex: `python plot_hist.py hist` for `hist.npy`.  
  a. Metrics: Mean score, Mean score (last 30 epochs), SD score, SD score (last 30 epochs), High score, High score epoch.  
