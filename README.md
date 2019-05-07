# cs181-p4
Practical 4: Reinforcement Learning for CS 181.  
Team Members: Vincent Li, Harry Fu, Nathan Zhao

## Instructions  
1. Run agent with `python Q_learn.py`.  
2. Change baseline hyperparameter and feature configurations in `Q_learn.py`.  
  a. Change hyperparameters in lines 13-18.  
  b. Change features in `tuple_generate` function in lines 46-96.  
3. Change number of epochs in line 205 of `Q_learn.py` (number of epochs is 3rd parameter of `run_games` function, default value is 100).
4. **IMPORTANT**: Change name of `.npy` file for each trial (to avoid overwriting data) in line 208.
5. Plot and save histogram while calculating metrics with `python plot_hist.py NAME_OF_NPY_FILE`. Ex: `python plot_hist.py hist` for `hist.npy`.  
  a. Metrics: Mean score, Mean score (last 30 epochs), SD score, SD score (last 30 epochs), High score, High score epoch.  

## State Feature Selection Trials  
1. Trial 1  
  a. Baseline (leave file unchanged).  
2. Trial 2  
  a. Uncomment line 93 and comment lines 83-91 to change Discretized Monkey Velocity calculation.  
3. Trial 3  
  a. Uncomment line 93 and comment lines 83-91 to change Discretized Monkey Velocity calculation.  
  b. Uncomment lines 57-60 and comment lines 49-54 to change Horizontal Distance calculation.  
4. Trial 4  
  a. Uncomment line 93 and comment lines 83-91 to change Discretized Monkey Velocity calculation.  
  b. Uncomment line 73 and comment lines 66-70 to change Danger Indicator calculation.  
5. Trial 5  
  a. Uncomment line 93 and comment lines 83-91 to change Discretized Monkey Velocity calculation.  
  b. Uncomment lines 76-80 and comment lines 66-70 to change Danger Indicator calculation.  
