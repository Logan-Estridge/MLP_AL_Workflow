Starting from the primary script `active_learning.sh` located in the `ROOT_d`. The first function defines and exports some important variables for the active learning cycle: 
```bash
set_plan() {                                         
  start_AL_iteration=0                                       
  final_AL_iteration=8                              
  export lower_limit=0.05 # QbC structure selection (Max devi f; ev/Angstrom)                                         
  export upper_limit=0.25 # QbC structure selection (Max devi f; ev/Angstrom)                                         
  export polymorph="ALPHA_I"                                      
  export version="v.11.updated-AL-workflow"
  AL_log="${polymorph}_AL.log"
}
```
These are the starting and final iteration of active learning you want to perform, the lower and upper limits for structure selection according to the query by committee method in the exploration step, the name of the current polymorph of interest, and the current version of the active learning workflow. Finally, this function also defines the name of the active learning log file for the given polymorph.

The next function, `define_directories`, simply defines and exports the locations of important landmark directories, and creates/updates the log file for the given polymorph. Adjust the location of the `ROOT_d` accordingly:
```bash
define_directories() {                                                                      
  export ROOT_d="/storage/nas_scr/shared/groups/valsson/Logan-Runs/PABA/PABA-MLP/${version}"                                         
  export LABEL_d="${ROOT_d}/labeling"                                         
  export TRAIN_d="${ROOT_d}/training"                                         
  export EXPLORE_d="${ROOT_d}/exploration"
  cd ${ROOT_d} && touch ${AL_log}
}    
```

The next function is the heart of the active learning workflow:
```bash
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
    echo "COMPLETED ITERATION ${iteration}; POLYMORPH ${polymorph}" >> ${AL_log}
  done                                                                          
}      
```
Based on the above values of `start` and `final_AL_iteration` it defines a `for` loop that iteratively goes through the active learning cycle from the `start` to the `final_AL_iteration`. E.g. if `start_AL_iteration = 0` and `final_AL_iteration = 5` it will perform the `for` loop for values of `iteration` corresponding to `0, 1, 2, 3, 4, and 5`. The first thing the `for` loop does is export the value of the current `iteration`, so that any subsequent scripts depending on this variable can call the correct current `iteration` value. It also defines the location of some directories often referenced in subsequent scripts which depend on the value of the current `iteration`. Next, it prints some simple information to the log file. Then, it moves into the labeling directory and executes `labeling.sh`, which performs all of the labeling (both training and validation sets) for any arbitrary iteration of active learning. This script is designed such that it will not finish until all of the labeling for the given iteration is completed. 
- Note: the `||` is the bash language logical OR operator. What it means is: if the previous command fails, execute the following command. Thus, if `cd ${LABEL_d}` was not able to be performed, that means it was probably misdefined in the above `set_plan` function.
- Note 2: `exit 1` means to exit the current script (`active_learning.sh`) with an exit code of `1`. This exit code means that the script has failed for any generic reason, the prior `echo` statement clarifies the reason. 

Once all the labeling for the given iteration is finished, the `for` loop will move into the training directory and execute `training.sh` 
- Note 3: if the training for a given iteration did not finish, uncomment `sh training-restart.sh` and comment `sh training.sh`. Comment other parts of `active_learing.sh` as necessary and set both `start` and `final_AL_iteration` to the value of the current iteration, and run the `active_learning.sh` script.

Likewise, this script (`training.sh`) will not complete until all the training is done for a given iteration. Finally, the `for` loop moves into the exploration directory and executes `exploration.sh`. As you can guess, this script is designed with exactly the same principles as the last two. Once `exploration.sh` is completed, the structures between the values of `lower_limit` and `upper_limit` defined in `set_plan` are extracted and sent to the next iteration's labeling directory for labeling. Thus the cycle may be repeated as many times as needed. 

For detailed explanations of `labeling.sh`, `training.sh`, and `exploration.sh` see the following:
- labeling.md
- training.md
- exploration.md

The last part of the `active_learning.sh` script simply executes the above three described functions in order, and finally this function is executed:
```bash
auto_AL() {         
  set_plan       
  define_directories
  active_learning   
}      

auto_AL
```
Thus, kicking off the automatic active learning cycle. 
