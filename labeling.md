# Directory Structure:
```bash
- ROOT_d/
	- labeling/
		- labeling.sh
		- scripts/
			- gen_QE_inputs.sh
			- ml_data.sh
			- qe_Gen-Batch-Scripts-better-CPU.py
			- QE_input_gen_from_gro_ecut-100.py
```
# Full Script: labeling.sh
```bash
#!/bin/bash

define_directories() {
  export TRAINING_SYSTEMS_d="${LABEL_TRAINING_d}/systems"
  export VALIDATION_SYSTEMS_d="${LABEL_VALIDATION_d}/systems"
}

gen_structures_from_Gromacs() {
  date
  echo "Generating QE input scripts from GROMACS dumped trajectories."
  cd ${LABEL_d}/scripts || { echo "labeling/scripts directory not found."; exit 1; }
  sh gen_QE_inputs.sh
}

label_training() {
  slurm_ids=()
  finished_jobs=()
  cancelled_jobs=()

  date
  echo "Starting labeling for the training set; polymorph ${polymorph}; iteration ${iteration}."
  cd ${TRAINING_SYSTEMS_d} || { echo "Training systems directory not found."; exit 1; }
}

label_validation() {
  slurm_ids=()
  finished_jobs=()
  cancelled_jobs=()

  date
  echo "Starting labeling for the validation set; polymorph ${polymorph}; iteration ${iteration}."
  cd ${VALIDATION_SYSTEMS_d} || { echo "Validation systems directory not found."; exit 1; }
}

perform_labeling() {
  python qe_Gen-Batch-Scripts-better-CPU.py

  for batch_job in $(ls batch_job_*.sh); do
    slurm_id=$(sbatch ${batch_job} | awk '{print $4}')
    slurm_ids+=(${slurm_id})
  done
}


wait_jobs_done() {
  while : ; do
    all_done=false
    for slurm_id in "${slurm_ids[@]}"; do
      state=$(sacct -j $slurm_id --format=State%20 -P | grep -E 'COMPLETED|CANCELLED' | awk 'NR < 2 {print $1}')
      if [ "$state" == "COMPLETED" ]; then
        if [[ ! " ${finished_jobs[@]} " =~ " ${slurm_id} " ]]; then
          finished_jobs+=($slurm_id)
        fi                                                                                                                                                                                               
      elif [ "$state" == "CANCELLED" ]; then
        if [[ ! " ${cancelled_jobs[@]} " =~ " ${slurm_id} " ]]; then
          cancelled_jobs+=(${slurm_id})
        fi
      fi
    done

    if [ $(( ${#finished_jobs[@]} + ${#cancelled_jobs[@]} )) -eq ${#slurm_ids[@]} ]; then
      all_done=true
    fi

    if [ ${all_done} = true ]; then
      break
    fi
    sleep 10                                                                                                                                                                                             
  done
}

echo_training() {
  date
  echo "Completed labeling for the training set; polymorph ${polymorph}; iteration ${iteration}."
}

echo_validation() {
  date
  echo "Completed labeling for the validation set; polymorph ${polymorph}; iteration ${iteration}."
  echo "Completed ALL labeling for polymorph ${polymorph}; iteration ${iteration}."
}

create_ml_data_training() {
  sh ml_data.sh || { echo "Training systems did not finish labeling."; exit 1; }
}

create_ml_data_validation() {
  sh ml_data.sh || { echo "Validation systems did not finish labeling."; exit 1; }
}

labeling_sh() {
  define_directories

  if [ ${iteration} = 0 ]; then
    gen_structures_from_Gromacs
  fi

  label_training
  perform_labeling
  wait_jobs_done
  create_ml_data_training
  echo_training

  label_validation
  perform_labeling
  wait_jobs_done
  create_ml_data_validation
  echo_validation
}

labeling_sh
```

## Outline
The structure of the labeling script is clear from the last function, `labeling_sh`, which defines the order and conditions by which the other functions in the script are run:
```bash
labeling_sh() {
  define_directories

  if [ ${iteration} = 0 ]; then
    gen_structures_from_Gromacs
  fi

  label_training
  perform_labeling
  wait_jobs_done
  create_ml_data_training
  echo_training

  label_validation
  perform_labeling
  wait_jobs_done
  create_ml_data_validation
  echo_validation
}

labeling_sh
```
First, the function defining the directories is run, then if the iteration is equal to 0, the function to generate Quantum Espresso input files from the `*.gro` files you placed in `ROOT_d/labeling/${polymorph}/initial_confs/*/`  will be run. Otherwise, the script will jump into a series of functions designed to label the training set, followed by a series to label the validation set. 

## Details
First, the directories are defined:
```bash
define_directories() {
  export TRAINING_SYSTEMS_d="${LABEL_TRAINING_d}/systems"
  export VALIDATION_SYSTEMS_d="${LABEL_VALIDATION_d}/systems"
}
```
Recall that the variables `LABEL_TRAINING_d` and `LABEL_VALIDATION_d` were exported in `active_learning.sh`, and as this labeling script is being run in a subshell of `active_learning.sh` we are free to call these variables without defining them again. 

Next, if the `${iteration} = 0`, then the function to generate QE input files from GROMACS extracted structures, `*.gro` files, will be executed:
```bash
gen_structures_from_Gromacs() {
  date
  echo "Generating QE input scripts from GROMACS dumped trajectories."
  cd ${LABEL_d}/scripts || { echo "labeling/scripts directory not found."; exit 1; }
  sh gen_QE_inputs.sh
}
```
This runs the script `gen_QE_inputs.sh`:
### gen_QE_inputs.sh
```bash
#!/bin/bash                                                                                                                                                                                              

create_QE_inputs() {                                         
  mkdir -p ${TRAINING_SYSTEMS_d} ${VALIDATION_SYSTEMS_d}                                         
  python QE_input_gen_from_gro_ecut-100.py                                         
}                                         

copy_useful_scripts() {                                         
  scripts_to_copy=("ml_data.sh" "get_raw.py" "qe_Gen-Batch-Scripts-better-CPU.py")                                         

  for script in "${scripts_to_copy[@]}"; do                                         
    cp ${LABEL_d}/scripts/${script} ${TRAINING_SYSTEMS_d}                                         
    cp ${LABEL_d}/scripts/${script} ${VALIDATION_SYSTEMS_d}                                         
  done                                         
}                                         

gen_QE_inputs() {                                         
  create_QE_inputs                                         
  copy_useful_scripts                                         
}                                         

gen_QE_inputs     
```
which creates the `TRAINING_SYSTEMS_d` and `VALIDATION_SYSTEMS_d` directories defined in `labeling.sh`, runs the python script `QE_input_gen_from_gro_ecut-100.py`, and copies over some other useful scripts to the training and validation systems directories. The python script that is being run here is a simple use of the atomic simulation environment (ASE):
### QE_input_gen_from_gro_ecut-100.py
```python
#!/home/lre0058/.conda/envs/deepmd/bin/python

import os
import re
import subprocess

from ase.calculators.espresso import Espresso
from ase.io import read, write

# Specify the pseudopotentials for the elements
pseudopotentials = {
    "H": "H_ONCV_PBE-1.2.upf",
    "C": "C_ONCV_PBE-1.2.upf",
    "N": "N_ONCV_PBE-1.2.upf",
    "O": "O_ONCV_PBE-1.2.upf",
}

## INPUT PARAMETERS
kpoints = "gamma"
offset = (0, 0, 0)

# Set up the input parameters for QE calculation
input_qe = {
    "calculation": "scf",
    "outdir": "./",
    "pseudo_dir": "/home/lre0058/pseudopotentials/NC/",
    "prefix": "PABA",
    "tprnfor": True,
    "tstress": True,
    "disk_io": "none",
    "system": {
        "ecutwfc": 100,
        "input_dft": "PBE",
        "vdw_corr": "grimme-d3",
    },
    "electrons": {"mixing_beta": 0.5, "electron_maxstep": 1000},
}

polymorph = os.getenv("polymorph")
LABEL_d = os.getenv("LABEL_d")
TRAINING_SYSTEMS_d = os.getenv("TRAINING_SYSTEMS_d")
VALIDATION_SYSTEMS_d = os.getenv("VALIDATION_SYSTEMS_d")

training_confs = f"{LABEL_d}/{polymorph}/initial_confs/training"
validation_confs = f"{LABEL_d}/{polymorph}/initial_confs/validation"
                    
output_training = f"{TRAINING_SYSTEMS_d}"
output_validation = f"{VALIDATION_SYSTEMS_d}"

# Read from .gro / write QE input files             
for file_name in os.listdir(training_confs):            
    if file_name.endswith(".gro"):                                                                
        match = re.search(r"(\d{5})", file_name)
        if match:                                                   
            file_number = f"{int(match.group(1)):05}"
            gro_file_name = (
                f"PABA_N32_NPT-Produc_CRpcoupl_300K.pbc-whole.{file_number}ps.gro"    
            )                            
            PABA_structure = read(f"{training_confs}/{gro_file_name}")                                                                                                                                   
                                                                                     
            # Write the input file for QE calculation using ASE's write() function
            qe_input_file_name = f"pw-PABA_{file_number}ps.in"
            write(                                        
                os.path.join(output_training, qe_input_file_name),
                PABA_structure,
                format="espresso-in",
                input_data=input_qe,
                pseudopotentials=pseudopotentials,
                kpts=kpoints,
                tstress=True,
                tprnfor=True,
            )

for file_name in os.listdir(validation_confs):
    if file_name.endswith(".gro"):
        match = re.search(r"(\d{5})", file_name)
        if match:
            file_number = f"{int(match.group(1)):05}"
            gro_file_name = (
                f"PABA_N32_NPT-Produc_CRpcoupl_300K.pbc-whole.{file_number}ps.gro"
            )
            PABA_structure = read(f"{validation_confs}/{gro_file_name}")

            # Write the input file for QE calculation using ASE's write() function
            qe_input_file_name = f"pw-PABA_{file_number}ps.in"
            write(
                os.path.join(output_validation, qe_input_file_name),
                PABA_structure,
                format="espresso-in",
                input_data=input_qe,
                pseudopotentials=pseudopotentials,
                kpts=kpoints,
                koffset=offset,
                tstress=True,
                tprnfor=True,
            )
```
Importantly, change the value of the `gro_file_name` as needed to match the naming scheme of your extracted `*.gro` structures. Also to note, the python function `os.getenv()` allows us to retrieve any bash environmental variables that we may have defined earlier and use them in this script.
### Back to labeling.sh
Going back to the main `labeling.sh` script, the next two functions have incredibly similar structure:
```bash
label_training() {
  slurm_ids=()
  finished_jobs=()
  cancelled_jobs=()

  date
  echo "Starting labeling for the training set; polymorph ${polymorph}; iteration ${iteration}."
  cd ${TRAINING_SYSTEMS_d} || { echo "Training systems directory not found."; exit 1; }
}

label_validation() {
  slurm_ids=()
  finished_jobs=()
  cancelled_jobs=()

  date
  echo "Starting labeling for the validation set; polymorph ${polymorph}; iteration ${iteration}."
  cd ${VALIDATION_SYSTEMS_d} || { echo "Validation systems directory not found."; exit 1; }
}
```
Their purpose is simply to initialize the bash arrays for the slurm ids, the finished jobs, and the cancelled jobs, and then to `cd` either into the training or validation systems directory.

The next function is very important, as it defines the main labeling procedure:
```bash
perform_labeling() {
  python qe_Gen-Batch-Scripts-better-CPU.py

  for batch_job in $(ls batch_job_*.sh); do
    slurm_id=$(sbatch ${batch_job} | awk '{print $4}')
    slurm_ids+=(${slurm_id})
  done                                                                                                                                                                                                   
}
```
First, it runs a python script which defines the creation of batch slurm jobs for the labeling process, and then it submits those batch jobs while capturing the slurm ids using `awk` and adding those ids to the (previously empty) bash array for the slurm ids.

The python script which creates batch slurm jobs is as follows:
### qe_Gen-Batch-Scripts-better-CPU.py
```python
import os
import subprocess
import sys

# CHANGE PARTITION HERE
PARTITION = "short.36" # <====
MAX_PARALLEL_JOBS = 4 # Change the number of parallel jobs per batch script if desired
                      # more parallel jobs = less cores per job and vice versa.

####################################################################################################
# Defining the batch scripts
####################################################################################################

def create_batch_script(start_job, end_job, cores_per_job, MAX_PARALLEL_JOBS):
    """Create a batch script for a given job range."""
    script_content = f"""#!/bin/bash
#
#SBATCH --job-name=qe_{start_job}-{end_job}
#SBATCH --output=check_batch_cpu_{start_job}-{end_job}.out
#SBATCH --error=check_batch_cpu_{start_job}-{end_job}.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={cores_per_job * MAX_PARALLEL_JOBS}
#SBATCH --mail-user=LoganEstridge@my.unt.edu
#SBATCH --mail-type=ALL
#SBATCH -p {PARTITION}
#

# Load modules
module purge
module load QE/7.1/imkl_gcc12.2_ompi4.1.4

export start_job={start_job}
export end_job={end_job}

# The batch script(s) main content
for i in $(ls pw-*.in | sed -n "${{start_job}},${{end_job}}p"); do
    input_file="$i"
    output_file="${{input_file%.in}}.out"

    echo "Starting job "$i""

    mpirun -np {cores_per_job} pw.x -input "$input_file" > "$output_file" &

    # Ensure only a max of max_parallel_jobs are running at the same time
    while (( $(jobs -r | wc -l) >= {MAX_PARALLEL_JOBS} )); do
        echo "Current number of parallel jobs: $(jobs -r | wc -l)"
        wait -n
    done
done
wait
"""

    # Write the script to a file                                                                                                                                                                         
    script_filename = f"batch_job_{start_job}-{end_job}.sh"
    with open(script_filename, "w") as script_file:
        script_file.write(script_content)

    return script_filename

####################################################################################################
# Defining the parameters for submitting the batch scripts
####################################################################################################

def get_total_jobs(): 
    result = subprocess.run(
        """ls pw*in | wc -l""",
        shell=True, capture_output=True, text=True
    )
    return int(result.stdout.strip())

def get_max_cpu_cores_per_user():  # <-- Runs 'partinfo' and returns the current 'max CPU cores per user' value for the PARTITION                                                                        
    result = subprocess.run(
        "partinfo | grep " + f"{PARTITION}" + " | awk '{print $3 }'",
        shell=True, capture_output=True, text=True
    )
    return int(result.stdout.strip())

def get_max_jobs_per_user():  # <-- Runs 'partinfo' and returns the current 'max Jobs per user' value for the PARTITION
    result = subprocess.run(
        "partinfo | grep " + f"{PARTITION}" + " | awk '{print $2 }'",
        shell=True, capture_output=True, text=True
    )
    return int(result.stdout.strip())

def get_cpu_cores_per_partition():  # <-- Runs 'scontrol show part {PARTITION}' and returns the number 
                                    # of cores per node for the selected PARTITION
    result = subprocess.run(
        "scontrol show part " + f"{PARTITION}" + " | grep MaxCPUsPerNode | awk '{print substr($0, length($0)-1, 2)}'",
        shell=True, capture_output=True, text=True
    )
    return int(result.stdout.strip())

total_jobs = get_total_jobs()
max_jobs_per_user = get_max_jobs_per_user()
max_cpu_cores_per_user = get_max_cpu_cores_per_user()
max_cpus_per_node = get_cpu_cores_per_partition()
jobs_per_script = get_total_jobs() // max_jobs_per_user
cores_per_job = max_cpus_per_node // MAX_PARALLEL_JOBS
ntasks_per_node = cores_per_job * MAX_PARALLEL_JOBS
number_parallel_cores = max_jobs_per_user * ntasks_per_node
running_jobs = jobs_per_script * max_jobs_per_user 

sys.stderr = open("ALL_BATCHES_CHECK.err", "w")

def check_cores():
    global max_jobs_per_user
    global number_parallel_cores
    global jobs_per_script
    global running_jobs
    
   while number_parallel_cores > max_cpu_cores_per_user:
        print(
            f"""**Error:** The total number of cores in use across all scripts ({number_parallel_cores}) exceeds the maximum allowed 
            ({max_cpu_cores_per_user}) for the PARTITION: ({PARTITION}). Current number of batch scripts: ({max_jobs_per_user}).""",
            file=sys.stderr,
        )

        max_jobs_per_user -= 1
        number_parallel_cores = max_jobs_per_user * ntasks_per_node                                                                                                                                      
        jobs_per_script = total_jobs // max_jobs_per_user
        running_jobs = jobs_per_script * max_jobs_per_user 

        print(
            f"""Decreasing the number of batch scripts by 1 to compensate: 
            New number of batch scripts: ({max_jobs_per_user}), 
            New number of total cores in use: ({number_parallel_cores}),
            New number of jobs per script: ({jobs_per_script}).""",
            file=sys.stderr,
        )

        if max_jobs_per_user <= 0:
            sys.exit("Job submission cancelled due to excessive core usage.")

    if jobs_per_script == 0:
        print(
            f"""**Error:** There are less total jobs ({total_jobs}) than the current number of scripts ({max_jobs_per_user}).""",
            file=sys.stderr,
        )
        jobs_per_script += 1
        max_jobs_per_user = total_jobs
        number_parallel_cores = max_jobs_per_user * ntasks_per_node
        running_jobs = jobs_per_script * max_jobs_per_user 
        print(
            f"""Setting the number of total jobs equal to the number of scrtipts:
            New jobs per script: {jobs_per_script}
            New number of scripts: {max_jobs_per_user}""",
            file=sys.stderr,
        )

    while running_jobs < total_jobs:
        print(
            f"""**Error:** There are less running jobs ({running_jobs}) than the number of total jobs ({total_jobs}).""",
            file=sys.stderr,
        )
        jobs_per_script += 1
        running_jobs = jobs_per_script * max_jobs_per_user 
        print(
            f"""Increasing the number of jobs per script to compensate:
            New number of jobs per script: {jobs_per_script}
            New number of running jobs: {running_jobs}""",
            file=sys.stderr,
        )

check_cores()

sys.stdout = open("ALL_BATCHES_CHECK.out", "w")

print(
    f"""Total jobs: {total_jobs}, Number of batch scripts: {max_jobs_per_user}, Cores per job: {cores_per_job},
    Jobs per script: {jobs_per_script}, Max parallel jobs per script: {MAX_PARALLEL_JOBS},
    ntasks-per-node: {ntasks_per_node}, Total parallel cores: {number_parallel_cores},
    Partition: {PARTITION}, Max CPUs per node: {max_cpus_per_node},
    Max CPU cores per user: {max_cpu_cores_per_user}""",
    file=sys.stdout,
)

####################################################################################################
# Defining the submission of the batch scripts
####################################################################################################

def submit_batch_scripts(total_jobs, cores_per_job, jobs_per_script, MAX_PARALLEL_JOBS):
    """Submit batch scripts based on the total number of jobs."""

    for i in range(max_jobs_per_user):
        start_job = i * jobs_per_script + 1
        end_job = min((i + 1) * jobs_per_script, total_jobs)
        batch_script = create_batch_script(start_job, end_job, cores_per_job, MAX_PARALLEL_JOBS)

####################################################################################################
# Submits the batch scripts with the desired parameters
####################################################################################################

submit_batch_scripts(total_jobs, cores_per_job, jobs_per_script, MAX_PARALLEL_JOBS)
```
This script is quite complicated, but its purpose is very simple, and that is to break up any given number of DFT calculation jobs into batches and to create slurm batch submission scripts that calculate each of these batches, such that near the maximum amount of total resources available is being used until all of the jobs are done. The script accomplishes this goal by collecting certain key information such as:
- the total number of DFT jobs to calculate: `ls pw*in | wc -l`
- the current maximum number of "jobs per user" and "cpu cores per user" for the selected partition, as defined by running `partinfo` in the shell
With this information, the script sets the values of certain key variables using some (definitely imperfect, but good enough) math in the function `check_cores()`. Many of these key variables are used in the function `create_batch_script()`, which defines the content of the slurm batch scripts, and ensures only a maximum of a certain number of jobs are running at any given time (so as to not overload the computational nodes). Finally, the function `submit_batch_scripts()` actually creates the batch scripts so that they can be submitted by `labeling.sh`. 

### Back to labeling.sh (again)
The next function will appear in a similar form in both [[exploration]].sh and [[training]].sh. In this case, it is responsible for holding the script from finishing its execution until all of the labeling is completed. It does so by taking advantage of the `sacct` function in the shell, which can monitor job status for a given slurm id:
```bash
wait_jobs_done() {    
  while : ; do    
    all_done=false    
    for slurm_id in "${slurm_ids[@]}"; do    
      state=$(sacct -j $slurm_id --format=State%20 -P | grep -E 'COMPLETED|CANCELLED' | awk 'NR < 2 {print $1}')    
      if [ "$state" == "COMPLETED" ]; then    
        if [[ ! " ${finished_jobs[@]} " =~ " ${slurm_id} " ]]; then    
          finished_jobs+=($slurm_id)    
        fi    
      elif [ "$state" == "CANCELLED" ]; then    
        if [[ ! " ${cancelled_jobs[@]} " =~ " ${slurm_id} " ]]; then    
          cancelled_jobs+=(${slurm_id})    
        fi    
      fi    
    done    
    
    if [ $(( ${#finished_jobs[@]} + ${#cancelled_jobs[@]} )) -eq ${#slurm_ids[@]} ]; then    
      all_done=true    
    fi                                                                                                                                                                                                                                                                  
    
    if [ ${all_done} = true ]; then    
      break    
    fi    
    sleep 10    
  done    
}    
```
According to `sacct`, if the sum of the length of the `finished_jobs` and `cancelled_jobs` arrays are equal to the length of the `slurm_ids` array, then the `all_done` condition will be met and the infinite while loop will be broken. Thus, the script can move on from this function to the next one.  

Finally, there are a couple of functions responsible for `echo`ing some information that don't need description. And the last important functions are the following:
```bash
create_ml_data_training() {
  sh ml_data.sh || { echo "Training systems did not finish labeling."; exit 1; }
}

create_ml_data_validation() {
  sh ml_data.sh || { echo "Validation systems did not finish labeling."; exit 1; }
}
```
They both run the exact same script:
### ml_data.sh
```bash
#!/bin/bash                                         

mkdir -p ../ml_data                                         

unconverged_frames=$( grep -L DONE. pw*.out )                                         
num_unconverged_frames=$( grep -L DONE. pw*.out | wc -l )                                         

if (( num_unconverged_frames > 0 )); then                                         
  echo "${num_unconverged_frames} structure(s) did not converge. Removing from the ml_data set..."                                         
  mkdir unfinished                                         
  for frame in ${unconverged_frames}; do                                         
    mv ${frame} unfinished/                                         
  done                                         
fi                                         

python get_raw.py                                         
execute_raw_to_set.sh                                         
mv *.raw ../ml_data/
mv set* ../ml_data/  
```
This script simply checks if any DFT calculations did not finish, and if there are any that did not, sends them to a different directory to separate them from the rest of the data. Then it runs a python script to convert the QE output data to `*.raw` format, and performs the DeePMD-kit function `execute_raw_to_set`, with the "nline per set" = 100 (see DeePMD-kit docs). Finally, it moves all of this "raw data" to a separate directory called `ml_data/`. 

### get_raw.py
```python
#!/home/lre0058/.conda/envs/deepmd/bin/python

import numpy as np                                         
import ase.io                                         
from ase.calculators.espresso import Espresso                                         
import subprocess                                         
import os                                         
import re                                         

def get_raws():                                         
    with open("coord.raw", "w") as file_coord, \                                         
        open("energy.raw", "w") as file_energy, \                                         
        open("force.raw", "w") as file_force, \                                         
        open("virial.raw", "w") as file_virial, \                                         
        open("box.raw", "w") as file_box, \                                         
        open("type.raw", "w") as file_type:                                         

        types_written=False                                         

        for file_name in os.listdir():                                         
            if file_name.endswith(".out"):                                         
                if file_name.startswith("pw"):                                         
                    try:                                         
                        conf=ase.io.read(file_name, format='espresso-out')                                         
                    except:                                         
                        print("Configuration " + file_name + " could not be read")                                         
                    else:                                         
                        try:                                         
                            conf.get_forces()                                         
                        except:                                         
                            print("Forces missing from file " + file_name)                                         
                        else:                                         
                            file_coord.write(' '.join(conf.get_positions().flatten().astype('str').tolist()) + '\n')                                         
                            file_energy.write(str(conf.get_potential_energy()) + '\n')                                         
                            file_force.write(' '.join(conf.get_forces().flatten().astype('str').tolist()) + '\n')                                         
                            file_virial.write(' '.join(conf.get_stress(voigt=False).flatten().astype('str').tolist()) + '\n')                                         
                            file_box.write(' '.join(conf.get_cell().flatten().astype('str').tolist()) + '\n')                                         
                            if (not(types_written)):                                         
                                types = np.array(conf.get_chemical_symbols())                                         
                                types[types=="H"]="0"                                         
                                types[types=="C"]="1"                                         
                                types[types=="N"]="2"                                         
                                types[types=="O"]="3"                                         
                                file_type.write(' '.join(types.tolist()) + '\n')                                         
                                types_written=True                                         

if __name__ == "__main__":                                         
    get_raws()                                                                                                                                                                                                  
```
This script is not the only way to convert QE output data to `*.raw` format. Also see the DeePMD-kit docs on the `dpdata` tool.
