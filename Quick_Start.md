`cd` into the root directory (`ROOT_d`) for the Active Learning workflow. This directory contains three folders corresponding to the three main processes involved in the active learning cycle: `labeling`, `training`, and `exploration` as defined above. Each of these directories contains a script corresponding to that process which performs all the tasks necessary to complete that process for a given iteration of active learning. These directories also contain folders with names such as `scripts`, `utils`, or `template`, etc., which contain helper scripts and template files to perform the tasks such as high-throughput DFT calculation, training an ensemble of DeePMD models, etc. 

Once you have generated the initial dataset and extracted the structures in `*.gro` format, `cd` into the `labeling` directory and create a directory with the name of your polymorph/system of interest, e.g. `ALPHA_I`, within this folder create a directory named `initial_confs`, and within this directory two folders named `training`, and `validation`. These two folders will serve as the location of the `*.gro` structures you extracted for the training and validation sets, respectively. Thus, hardlink the structures you set aside for the training set into the `initial_confs/training` directory, and likewise for the validation set: e.g. `ln some_location/training_set_structures/*.gro ROOT_d/labeling/ALPHA_I/initial_confs/training`. To be very clear, the basic outline of the directory structure at this point should appear as such:

```bash
ROOT_d/
	exploration/
	labeling/
		ALPHA_I/
			initial_confs/
				training/
					*.gro
				validation/
					*.gro
		scripts/
		labeling.sh
	training/
```

Next, take one of the structures from the training set, and convert that structure into LAMMPS topology format (`conf*.lmp`) using the Atomic Simulation Environment (ASE: https://wiki.fysik.dtu.dk/ase/) python module. Make sure to include the masses card, which ASE may not have included by default. E.g.: 
```lammps
Masses                                                                                                                         
1 1.008   # H                                                                                                                  
2 12.011  # C                                                                                                                  
3 14.007  # N                                                                                                                               
4 15.999  # O       
```
The first column is the atom I.D. (the same I.D. should be used here as in the DeePMD training. To be consistent, I name my elements from lightest to heaviest in numerical order from lowest to highest, as you can see above), with the second column being the mass. 

Take this LAMMPS topology file and name it `conf_${polymorph}_ASE.lmp` and place it into the template directory in the exploration folder. I.e.:
```bash
ROOT_d/
	exploration/
		template.PABA/
			conf_ALPHA_I_ASE.lmp
			... other files
	labeling/
	training/
```

Now all the preliminary setup to start the active learning cycle is done. `cd` back into the root directory and inspect the active learning script: `vi active_learning.sh`, it should appear something like the following:
```bash
#!/bin/bash                                         

set_plan() {                                         
  start_AL_iteration=0                                       
  final_AL_iteration=8                              
  export lower_limit=0.05 # QbC structure selection (Max devi f; ev/Angstrom)                                         
  export upper_limit=0.25 # QbC structure selection (Max devi f; ev/Angstrom)                                         
  export polymorph="ALPHA_I"                                      
  export version="v.11.updated-AL-workflow"
  AL_log="${polymorph}_AL.log"
}

define_directories() {                                                                      
  export ROOT_d="/storage/nas_scr/shared/groups/valsson/Logan-Runs/PABA/PABA-MLP/${version}"                                         
  export LABEL_d="${ROOT_d}/labeling"                                         
  export TRAIN_d="${ROOT_d}/training"                                         
  export EXPLORE_d="${ROOT_d}/exploration"
  cd ${ROOT_d} && touch ${AL_log}
}                                         

active_learning() {                                                      
  for iteration in $(seq ${start_AL_iteration} ${final_AL_iteration}); do                                         
    export iteration                                                                
    export LABEL_TRAINING_d="${LABEL_d}/${polymorph}/iter_${iteration}/training"    
    export LABEL_VALIDATION_d="${LABEL_d}/${polymorph}/iter_${iteration}/validation"

    echo "$(date)" >> ${AL_log}
    echo "STARTING ITERATION ${iteration}; POLYMORPH ${polymorph}" >> ${AL_log}

    cd ${LABEL_d} || { echo "labeling directory not found."; exit 1; }
    sh labeling.sh                                                                                             

    cd ${TRAIN_d} || { echo "training directory not found."; exit 1; }     
    sh training.sh                                                                                             
    # sh training-restart.sh                                               

    cd ${EXPLORE_d} || { echo "exploration directory not found."; exit 1; }
    sh exploration.sh

    cd ${ROOT_d}
    echo "$(date)" >> ${AL_log}
    echo "STARTING ITERATION ${iteration}; POLYMORPH ${polymorph}" >> ${AL_log}
  done                                                                          
}                                              

auto_AL() {         
  set_plan       
  define_directories
  active_learning   
}      

auto_AL
```

## The zen of the above script
The automatic active learning workflow takes advantage of two key properties of bash scripting: 1. it is sequential, i.e. it (generally) goes from top to bottom and executes and finishes any upper task before starting any lower task (thus, e.g. I can ensure all of the labeling finishes in `labeling.sh` before any training starts in `training.sh`); and 2. it allows for the "exportation" of variables. Exporting a variable technically means that the variable will be available to be called in any subshell created by the script the variable was exported in. What this practically means is that if I, for example, export a variable such as "`ROOT_d`" in the above script, then in any script which is run within the above script, such as `labeling.sh`, I can just call this variable in `labeling.sh` (e.g. `cd ${ROOT_d}`) without having to define it anywhere within `labeling.sh`. This strategy of exporting as many variables as possible makes the code of the entire workflow cleaner and more flexible. For example, by exporting the value of the `iteration` of active learning in the above script, I never have to define  it (and therefore change very often) in any of the scripts related to the labeling, training, or exploration processes.

## Practically
Starting from the initial dataset, the value of `start_AL_iteration` should be equal to `0`. For `final_AL_iteration`, a value of at least `3` to `5`, and up to `8`, or so, should be chosen. The reason is that you want to give the system at least a few cycles of active learning to see how its behavior develops, but not too many before changing the settings in the training or exploration to make the active learning more efficient and the MLP more accurate.

Set the location of the `ROOT_d` for the active learning workflow in the `define_directories` function. This includes the `version` variable, which is just the parent directory of `ROOT_d` defining the current development version of the active learning workflow (which should be defined in `set_plan`). Finally, define the polymorph in `set_plan`, which should have the same name as the directory you made in `ROOT_d/labeling/`. If all of these parameters are satisfactory, save and close the file, and run it with `sh active_learning.sh`. You should see a runnning printout on the screen corresponding to the processes being executed by the script. I.e., the date and time corresponding to the stage of active learning for the given polymorph and iteration. Once the active learning cycle finishes a given iteration (let's say `iteration_n`), then relevant plots for the model deviations can be found in `ROOT_d/exploration/${polymorph}/iter_n/*.png`. The most salient plot will be named `hist_plot_max_devi_f.png`, which shows the maximum deviations in the forces over the deep potential MD run in the exploration step between 4 DeePMD models trained in the training step, along with a histogram of the number of structures corresponding to the model deviation.

This plot will look something like the following:
![hist_iter_11](https://github.com/user-attachments/assets/3361dddd-f4e3-4b65-9dd1-e36ceefa7cbd)
