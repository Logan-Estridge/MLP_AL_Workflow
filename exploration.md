# Directory Structure:
```bash
- ROOT_d/
	- exploration/
		- exploration.sh
		- scripts/
			- ALL_MODEL_DEVI.py
			- ALL_MODEL_DEVI_HIST.py
			- get_all_labeling_inputs.sh
		- template.PABA/
			- CONF_${polymorph}_ASE.lmp
			- convert-to-QE-input.py
			- get_configurations.py
			- input.lmp
			- lammps.sub
			- save.vmd # saved visualization state for vmd (optional)
```
# Full Script: exploration.sh
```bash
#!/bin/bash

set_plan() {
  export INPUT="input.lmp"
  export CONF_LMP="conf_${polymorph}_ASE.lmp"
  export GRAPH_d="${TRAIN_d}/${polymorph}/iter_${iteration}/?"
  export TMPL_d="${EXPLORE_d}/template.PABA"
  export WRK_p="${EXPLORE_d}/${polymorph}/iter_${iteration}"
  mkdir -p ${WRK_p}

  date
  echo "Starting exploration for polymorph ${polymorph}; iteration ${iteration}"
}

set_dpmd_variables() {
  export T=280
  export out_freq=40
  export dump_freq=40
  export steps=12000
  export time_step=0.0005
  export temp_damp=0.05
  export dnn_list="graph-compress-1.pb graph-compress-2.pb graph-compress-3.pb graph-compress-4.pb"
  export press=1
  export press_damp=0.5
}

create_and_submit_work() {                                                                                                                                                                                                                                                                                                         
  scripts_to_copy=("ALL_MODEL_DEVI.py" "ALL_MODEL_DEVI_HIST.py" "ALL_MODEL_THERMO.py" "get_all_labeling_inputs.sh")

  for script in ${scripts_to_copy[@]}; do
    cp ${EXPLORE_d}/scripts/${script} ${WRK_p}
  done

  slurm_ids=()
  log_files=()
  finished_jobs=()
  cancelled_jobs=()

  for md_id in $(seq 10); do
    export md_id
    export random_seed=$((md_id * 100))
    export md_plan="md-random_seed-${random_seed}"
    export WRK_d="${WRK_p}/${md_plan}"
    rm -rf ${WRK_d}
    mkdir -p ${WRK_d}

   templates_to_copy=(
      "${CONF_LMP}"
      "${INPUT}"                                                                                                                               
      "lammps.sub"
      "convert-to-QE-input.py"
      "get_configurations.py" 
      "make_QE_input_files.sh"             
      "save.vmd"                           
    )          

    for template in ${templates_to_copy[@]}; do
      cp ${TMPL_d}/${template} ${WRK_d}
    done

    sed -i "s/<T>/$T/g; s/<random_seed>/${random_seed}/g; s/<out_freq>/${out_freq}/g; s/<dump_freq>/${dump_freq}/g; \
      s/<steps>/${steps}/g; s/<time_step>/${time_step}/g; s/<temp_damp>/${temp_damp}/g; s/<dnn_list>/${dnn_list}/g; \
      s/<press>/${press}/g; s/<press_damp>/${press_damp}/g; s/<conf_lmp>/${CONF_LMP}/g" "${WRK_d}"/"${INPUT}"

    ln -s ${GRAPH_d}/graph-compress-?.pb ${WRK_d}

    cd ${WRK_d}
    slurm_id=$(sbatch lammps.sub | awk '{print $4}')

    slurm_ids+=($slurm_id)
    log_files+=("${WRK_d}/_slurm-${slurm_id}.out")
  done
}

wait_jobs_done() {
  while : ; do
    all_done=false
    for slurm_id in "${slurm_ids[@]}"; do
      state=$(sacct -j $slurm_id --format=State%20 -P | grep -E 'COMPLETED|CANCELLED' | awk 'NR < 2 {print $1}')
      if [ "$state" == "COMPLETED" ]; then
        if [[ ! " ${finished_jobs[@]} " =~ " $slurm_id " ]]; then
          echo "Committee with slurm id ${slurm_id} has finished exploration."
          finished_jobs+=($slurm_id)
        fi
      elif [ "$state" == "CANCELLED" ]; then
        if [[ ! " ${cancelled_jobs[@]} " =~ " $slurm_id " ]]; then
          echo "Committee with slurm id ${slurm_id} has finished exploration."
          cancelled_jobs+=($slurm_id)
        fi
      fi
    done

    if [ $(( ${#finished_jobs[@]} + ${#cancelled_jobs[@]} )) -eq 10 ]; then
      all_done=true
    fi

    if [ $all_done = true ]; then
      break
    fi

    sleep 10
  done
}

wrap_up() {
  date
  echo "Exploration for polymorph ${polymorph}; iteration ${iteration} has completed."
}

perform_dpmd() {
  set_plan
  set_dpmd_variables
  create_and_submit_work
  wait_jobs_done
  wrap_up
}

perform_dpmd

generate_plots() {
  cd ${WRK_p}
  python ALL_MODEL_DEVI.py
  python ALL_MODEL_DEVI_HIST.py
}

extract_configurations() {
  date
  echo "Extracting structures for polymorph ${polymorph}; for iteration ${iteration} of active learning."

  cd ${WRK_p}
  sh get_all_labeling_inputs.sh

  date                                                                                                                                                                                                                                                                                                                             
  echo "Completed extracting structures for polymorph ${polymorph}; for iteration ${iteration} of active learning."
}                                                                                                                                                                                                                                                                                                                                  

post_processing() {
  generate_plots
  extract_configurations
}                                                                                                                                                                                                                                                                                                                                  
post_processing 
```

## Outline
The following two functions outline the exploration.sh script:
```bash
perform_dpmd() {
  set_plan
  set_dpmd_variables
  create_and_submit_work
  wait_jobs_done
  wrap_up
}

post_processing() {
  generate_plots
  extract_configurations
}  
```
- In `set_plan` some important variables are defined
- `set_dpmd_variables` defines the variables to be inserted into the LAMMPS input script
- `create_and_submit_work` creates the directories and submits a series of 10 MD simulations with different random seeds
- `wait_jobs_done` was explained in detail in [[4 - Journal/PABA MLP/Documentation/labeling|labeling]]
- `wrap_up` just `echo`s some info to the screen
- `generate_plots` makes plots like the one in [[MLP Automatic Active Learning Workflow]]
- Finally, `extract_configurations` extracts all of the structures between `lower_limit` and `upper_limit` as defined in `active_learning.sh` and sends them to the next iteration for labeling

## Details
`set_plan` and `set_dpmd_variables` are self explanatory.

The `create_and_submit_work` function is the heart of the exploration process. In the first part of the function, I simply copy over some scripts, and initialize some bash arrays:
```bash
 scripts_to_copy=("ALL_MODEL_DEVI.py" "ALL_MODEL_DEVI_HIST.py" "ALL_MODEL_THERMO.py" "get_all_labeling_inputs.sh")

  for script in ${scripts_to_copy[@]}; do
    cp ${EXPLORE_d}/scripts/${script} ${WRK_p}
  done

  slurm_ids=()
  log_files=()
  finished_jobs=()
  cancelled_jobs=()
```
The next part of the function begins a long `for` loop. This for loop is designed to submit a series of 10 MD simulations with different random seeds. This is accomplished by giving each MD run an id (`md_id`), and simply making the random seed a product of the `md_id`:
```bash
 for md_id in $(seq 10); do
    export md_id
    export random_seed=$((md_id * 100))
```
Next, I create the work directories, and copy over some useful scripts to each of these 10 directories:
```bash
  export md_plan="md-random_seed-${random_seed}"
    export WRK_d="${WRK_p}/${md_plan}"
    rm -rf ${WRK_d}  
    mkdir -p ${WRK_d}

   templates_to_copy=(
      "${CONF_LMP}"
      "${INPUT}"                                                                                                                               
      "lammps.sub"
      "convert-to-QE-input.py"
      "get_configurations.py" 
      "make_QE_input_files.sh"             
      "save.vmd"                           
    )                                                                                                                                                 

    for template in ${templates_to_copy[@]}; do
      cp ${TMPL_d}/${template} ${WRK_d}
    done
```
Then, I use the command-line tool `sed` to replace some placeholders in the LAMMPS input file with the variables defined in `set_dpmd_variables`:
```bash
   sed -i "s/<T>/$T/g; s/<random_seed>/${random_seed}/g; s/<out_freq>/${out_freq}/g; s/<dump_freq>/${dump_freq}/g; \
      s/<steps>/${steps}/g; s/<time_step>/${time_step}/g; s/<temp_damp>/${temp_damp}/g; s/<dnn_list>/${dnn_list}/g; \
      s/<press>/${press}/g; s/<press_damp>/${press_damp}/g; s/<conf_lmp>/${CONF_LMP}/g" "${WRK_d}"/"${INPUT}"
```
To save on storage I use symlinks to "copy" the compressed DeePMD models to the work directory:
```bash
    ln -s ${GRAPH_d}/graph-compress-?.pb ${WRK_d}
```
Finally, I submit each of the 10 MD simulations, while capturing the slurm ids, and also the log files:
```bash
    cd ${WRK_d}
    slurm_id=$(sbatch lammps.sub | awk '{print $4}')

    slurm_ids+=($slurm_id)
    log_files+=("${WRK_d}/_slurm-${slurm_id}.out")
  done
```

The LAMMPS submission script (`lammps.sub`) is as follows:
### lammps.sub
```bash
#!/bin/bash                                         
#                                         
#SBATCH --job-name=lammps                                         
#SBATCH -e _slurm-%j.out                                         
#SBATCH -o _slurm-%j.out                                         
#SBATCH --ntasks-per-node=18       ## Number of tasks per node                                         
#SBATCH --cpus-per-task=1          ## Number of cpus used per task                                         
#SBATCH --nodes=1                  ## Number of nodes to be used                                         
#SBATCH -p short.36                                         
#SBATCH --time=24:00:00                                         
#                                         

source /home/lre0058/deepmd-kit/bin/activate /home/lre0058/deepmd-kit/                                         
module load cuda11.8/toolkit/11.8.0                                         

NumMPI_PROC=18                                         
NumThreads=1                                         

INPUT=input.lmp                                                                                                                                
LAMMPS_BIN=lmp                                         

export OMP_NUM_THREADS=${NumThreads}                                         
export TF_INTRA_OP_PARALLELISM_THREADS=${NumThreads}                                         
export TF_INTER_OP_PARALLELISM_THREADS=1                                         


which ${LAMMPS_BIN}                                         
which mpirun                                          
env                                          

mpirun -np ${NumMPI_PROC} ${LAMMPS_BIN} -in ${INPUT}  
```
As you can see, it is a simple CPU slurm submission script, which runs `lmp -in input.lmp`. The LAMMPS input file is as follows:
### input.lmp
```bash
# Initialization
echo both
units metal
dimension 3
boundary p p p
atom_style atomic
pair_style deepmd <dnn_list> out_freq <out_freq>
restart <out_freq> restart.lmp restart2.lmp

# System Definition 
read_data <conf_lmp>

# Simulation Settings 
neigh_modify delay 10 every 1
pair_coeff * *                                                                                                                                                           

# Visualization
thermo <out_freq>
thermo_style custom step temp pe etotal epair emol press lx ly lz vol
fix thermo_print all print <out_freq> "$(step) $(temp) $(press) $(enthalpy) $(vol)" file thermo.txt screen no title "#step temp press enthalpy vol"
dump myDump all atom <dump_freq> pos.lammpstrj

# Production run in NPT
velocity all create <T> <random_seed> mom yes rot yes dist gaussian
fix 1 all nph iso <press> <press> <press_damp>
fix 2 all temp/csvr <T> <T> <temp_damp> <random_seed>
fix 3 all momentum 10000 linear 1 1 1 rescale
timestep <time_step>
run <steps> upto # 2 ps

write_data data.final
write_restart restart.lmp
```
Note the placeholder variables that will be replaced by the earlier `sed` expression. 

### Back to exploration.sh
The next important function to explain is:
```bash
generate_plots() {
  cd ${WRK_p}
  python ALL_MODEL_DEVI.py
  python ALL_MODEL_DEVI_HIST.py
}
```
which does as the name suggests. The first python script generates several scatterplots of all 10 MD runs for each of the categories of model deviations printed to the file `model_devi.out`. The second script is very similar, only also including a marginal histogram. 
### ALL_MODEL_DEVI.py
```python
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os
import numpy as np

iteration = int(os.getenv("iteration"))
# lower_limit = float(os.getenv("lower_limit"))
# upper_limit = float(os.getenv("upper_limit"))

# List of directories containing model_devi.out files
directories = glob.glob('md-*/')

# Function to extract the random_seed number from the directory name
def get_random_seed(directory):
    match = re.search(r'random_seed-(\d{3,4})', directory)                                                                                                                                                                                                                                                      
    if match:
        return int(match.group(1))
    return float('inf')  # Return a high number if no match is found

# Sort directories by the random_seed number
directories.sort(key=get_random_seed)

# Function to translate column names into descriptive titles
def translate_title(column):
    title_parts = {
        'max_': 'Maximum ',
        'min_': 'Minimum ',
        'avg_': 'Average ',
        'devi': 'Deviations',
        '_f': ' in the Forces',
        '_v': ' in the Virials'
    }

    title = column
    for key, value in title_parts.items():
        title = title.replace(key, value)
    return title

# Columns to plot
columns_to_plot = ['max_devi_f', 'min_devi_f', 'avg_devi_f', 'max_devi_v', 'min_devi_v', 'avg_devi_v']

# Loop through each column to create a separate overlay plot
for column in columns_to_plot:
    plt.figure(figsize=(8,8))

    for directory in directories:
        # Construct the file path
        file_path = f"{directory}/model_devi.out"

        match = re.search(r'random_seed-\d{3,4}', directory)
        if match:
            random_seed = match.group()
            legend_label = f"MD random seed = {random_seed.split('-')[-1]}"
        else:
            legend_label = directory

        # Open the file and read the first line for headers
        with open(file_path, 'r') as file:
            header = file.readline().strip()[1:]  # Remove the '#' character
            # Read the rest of the data using the adjusted headers
            data = pd.read_csv(file, delim_whitespace=True, names=header.split())

        # Column 1 as the X axis (steps)
        x = data.iloc[:, 0]
        x_label = data.columns[0]
        x_data_scaled = [xi * 0.0005 for xi in x]

        # Find the index of the current column to plot
        y_index = data.columns.get_loc(column)
        y = data.iloc[:, y_index]

        # Add the series to the plot
        plt.plot(x_data_scaled, y, '.', markersize=3, label=legend_label)

    plt.xlabel('Simulation Time (ps)')
    plt.ylabel(f'{translate_title(column)} (eV / $\mathregular{{\AA}}$)' if "_f" in column else f'{translate_title(column)} (eV)')
    plt.xticks(np.arange(0, 6.5, 0.5))
    plt.xlim(0,6)
    plt.yscale('log')

    if column == 'max_devi_f':
        plt.ylim(0.01, 100)
        plt.axhspan(ymin=0.05, ymax=0.25, color='lightgreen', alpha=0.5, label="Up to 300 Random Structures\nin This Range Selected\nfor Active Learning")

    plt.title(f"Iteration {iteration}\nQuery by Committee Active Learning: 4 DeePMD Models")

    # Position the legend below the plot in two columns
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), markerscale=3, ncol=2)

    # Adjust layout to accommodate the title and legend
    plt.tight_layout(rect=[0, 0, 1, 1])                                                                                                                                                                                                                                                                         
    # Save the overlay plot for the current column
    plt.savefig(f'devis_plot_{column}.png', dpi=1000)
    plt.close()     
```

### ALL_MODEL_DEVI_HIST.py
```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from matplotlib.legend_handler import HandlerPatch
import glob
import re
import os
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors

# List of directories containing model_devi.out files
directories = glob.glob('md-*/')
lower_limit = 0.05
upper_limit = 0.25
x_max = 6                                                                                                                                                                                                                                                                                                                                                                                   
x_max_plus = x_max + 0.5
num_structures = 300
iteration = int(os.getenv("iteration"))

# Function to translate column names into descriptive titles
def translate_title(column):
    title_parts = {
        'max_': 'Maximum ',
        'min_': 'Minimum ',
        'avg_': 'Average ',
        'devi': 'Deviations',
        '_f': ' in the Forces',
        '_v': ' in the Virials'
    }

    title = column
    for key, value in title_parts.items():
        title = title.replace(key, value)
    return title

# Columns to plot
columns_to_plot = ['max_devi_f', 'min_devi_f', 'avg_devi_f', 'max_devi_v', 'min_devi_v', 'avg_devi_v']

for column in columns_to_plot:
    combined_x_data = []
    combined_y_data = []

    for directory in directories:
        # Construct the file path
        file_path = f"{directory}/model_devi.out"

        # Open the file and read the first line for headers
        with open(file_path, 'r') as file:
            header = file.readline().strip()[1:]  # Remove the '#' character
            # Read the rest of the data using the adjusted headers
            data = pd.read_csv(file, delim_whitespace=True, names=header.split())

        # Column 1 as the X axis (steps)
        x = data.iloc[:, 0]
        x_label = data.columns[0]
        x_data_scaled = [xi * 0.0005 for xi in x]

        # Find the index of the current column to plot
        y_index = data.columns.get_loc(column)
        y = data.iloc[:, y_index]

        # Append data to combined arrays
        combined_x_data.extend(x_data_scaled)
        combined_y_data.extend(y)

    # Create a DataFrame with the combined data for use with seaborn
    plot_data = pd.DataFrame({'x': combined_x_data, 'y': combined_y_data})

    # Create the main plot with a scatterplot and a marginal histogram on the y-axis
    g = sns.JointGrid(data=plot_data, x='x', y='y', space=0, ratio=5)

    # Scatterplot in the main plot area
    g.plot_joint(sns.scatterplot, s=5, color='b')

    # Marginal histogram on the y-axis
    g.ax_marg_x.remove()

    # Create logarithmically spaced bins
    bins = np.logspace(np.log10(0.005), np.log10(50), 100)

    hist_counts, y_edges = np.histogram(plot_data['y'], bins=bins)
    hist_colors = plt.cm.viridis(hist_counts / hist_counts.max())
    widths = np.diff(y_edges)
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    for count, y, width, color in zip(hist_counts, y_centers, widths, hist_colors):
        g.ax_marg_y.barh(y=y, width=count, height=width, color=color, edgecolor='none')

    # Add color bar for the marginal histogram
    norm = mcolors.Normalize(vmin=0, vmax=hist_counts.max())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    g.fig.colorbar(sm, ax=g.ax_marg_y, orientation='vertical')

    # Set labels and title
    g.ax_joint.set_xlabel('Simulation Time (ps)')
    g.ax_joint.set_ylabel(f'{translate_title(column)} (eV / $\mathregular{{\AA}}$)' if "_f" in column else f'{translate_title(column)} (eV)')
    g.ax_joint.set_xticks(np.arange(0, x_max_plus, 0.5))
    g.ax_joint.set_xlim(0, x_max)
    g.ax_joint.set_yscale('log')

    if column == 'max_devi_f':
        g.ax_joint.set_ylim(0.005, 50)
        g.refline(y=lower_limit)
        g.refline(y=upper_limit)
        axhspan = g.ax_joint.axhspan(ymin=lower_limit, ymax=upper_limit, color='lightgreen', alpha=0.5)

        handles, labels = g.ax_joint.get_legend_handles_labels()
        patch = mpatches.Patch(color='lightgreen', alpha=0.5, label=f"Up to {num_structures} Random Structures\nin this Range Selected\nfor Active Learning")
        handles.append(patch)
        g.ax_joint.legend(handles=handles)                                                                                                                                                                                                                                                                                                                                                  
    g.ax_joint.set_title(f"Iteration {iteration}\nQuery by Committee Active Learning: 4 DeePMD Models")                                                                                                                                                                                                                                                                                     
    # Adjust layout to accommodate the title and legend                                                                                                                                                                                                                                                                                                                                     
    g.fig.set_size_inches(8, 8)
    plt.tight_layout(rect=[0, 0, 1, 1], pad=2.0)                                                                                                                                                                                                                                                                                                                                            
    # Save the overlay plot for the current column                                                                                                                                                                                                                                                                                                                                          
    plt.savefig(f'hist_plot_{column}.png', dpi=1000)
    plt.close()
```

### Back to exploration.sh (again)
The final important function is responsible for extracting structures for the next iteration of active learning:
```bash
extract_configurations() {
  date
  echo "Extracting structures for polymorph ${polymorph}; for iteration ${iteration} of active learning."

  cd ${WRK_p}
  sh get_all_labeling_inputs.sh

  date
  echo "Completed extracting structures for polymorph ${polymorph}; for iteration ${iteration} of active learning."
}
```
It runs the the following script:
### get_all_labeling_inputs.sh
```bash
#!/bin/bash

define_variables() {
  next_iteration=$((iteration + 1))
  export ecut="100"
  total_structures=0
}

extract_structures() {
  for d in md-*; do
    export random_seed=$(echo $d | grep -oE '[0-9]{3,4}')                                                                                                                                                                                                                                                                                                                                   
    # echo "Random seed for $d is $random_seed"

    pushd $d > /dev/null

    awk -v low="$lower_limit" -v up="$upper_limit" 'NR > 2 { if ($5 >= low && $5 < up) print $1 }' ./model_devi.out > timesteps

    if [ -d "extracted-confs" ]; then
      rm -r "extracted-confs"
    fi

    python get_configurations.py
    python convert-to-QE-input.py

    num_structures=$(ls ./extracted-confs/pw*.in 2> /dev/null | wc -l || echo 0)
    # echo "$num_structures structures for $d"
    total_structures=$((total_structures + num_structures))
    popd > /dev/null
  done

  echo "Total structures between $lower_limit and $upper_limit ev/Angstrom = $total_structures"
}

copy_structures_to_next_iteration_labeling_dir() {
  if [ -d "systems" ]; then
    rm -r "systems"
  fi

  mkdir systems
  for d in md-*; do
    ln $d/extracted-confs/pw*.in systems 2> /dev/null
  done

  training_dir="${LABEL_d}/${polymorph}/iter_${next_iteration}/training/systems"
  validation_dir="${LABEL_d}/${polymorph}/iter_${next_iteration}/validation/systems"

  if [ -d $training_dir ]; then
    rm -r $training_dir
  fi

  if [ -d $validation_dir ]; then
    rm -r $validation_dir
  fi

  mkdir -p $training_dir
  mkdir -p $validation_dir

  scripts_to_copy=("ml_data.sh" "get_raw.py" "qe_Gen-Batch-Scripts-better-CPU.py")

  for script in "${scripts_to_copy[@]}"; do
    cp ${LABEL_d}/scripts/${script} ${training_dir}
    cp ${LABEL_d}/scripts/${script} ${validation_dir}
  done

  pushd systems
  all_systems=($(ls $PWD | shuf))
  len_all_systems=${#all_systems[@]}

  if (( len_all_systems > 300 )); then
    num_systems_to_label=300
  elif (( len_all_systems < 30 )); then
    num_systems_to_label=0
  else
    num_systems_to_label=${len_all_systems}
  fi

  if (( num_systems_to_label == 0 )); then
    echo "Not enough structures in the model deviation window. Exiting..."
    exit 1
  fi

  systems_to_label=("${all_systems[@]:0:$num_systems_to_label}")

  num_training=$(( (num_systems_to_label * 90) / 100))
  num_validation=$(( num_systems_to_label - num_training ))

  echo "${num_training} structures sent to iteration ${next_iteration} training set for labeling"
  echo "${num_validation} structures sent to iteration ${next_iteration} validation set for labeling"

  for ((i=0; i<num_training; i++)); do
    ln "$PWD/${systems_to_label[i]}" "$training_dir" || \
      { echo "Next iteration's (${next_iteration}) training systems directory not found."; exit 1; }
  done

  for ((i=num_training; i<num_systems_to_label; i++)); do
    ln "$PWD/${systems_to_label[i]}" "$validation_dir" || \
      { echo "Next iteration's (${next_iteration}) validation systems directory not found."; exit 1; }
  done
  popd
}

get_all_labeling_inputs() {
  define_variables
  extract_structures
  copy_structures_to_next_iteration_labeling_dir
}

get_all_labeling_inputs
```
To summarize this script, it uses `awk` to extract all of the timesteps corresponding to structures with a maximum deviation in the force between `lower_limit` and `upper_limit` (as defined in `active_learning.sh`), printing these timesteps to a file called `timesteps`. Then, the python script `get_configurations.py` is run to extract the frames from the LAMMPS trajectory, and the python script `convert-to-QE-input.py` is run to convert these frames to Quantum Espresso input format, as in [[4 - Journal/PABA MLP/Documentation/labeling|labeling]]. After this, the QE inputs are shuffled, and a certain number of them are hardlinked over to the next iteration's `labeling/` directory. At this moment, I am `ln`-ing up to a maximum of 300 structures total, and a minimum of 30 total, with 90 % of the total going towards the next iteration's training set, and the other 10 % going towards the next iteration's validation set.
### get_configurations.py
```python
import os                                         

# Create the directory if it doesn't exist                                         
os.makedirs('extracted-confs', exist_ok=True)                                         

# Load the timesteps from the file                                         
with open('timesteps', 'r') as tfile:                                         
    timesteps = [int(line.strip()) for line in tfile]                                         

# Initialize variables                                         
current_timestep = None                                         
frames = []                                         
frame = []                                         
reading_atoms = False                                         
box_bounds = ""                                         
num_atoms = 0                                         
atom_types = ""                                         

# Open the trajectory file and process it
with open('pos.lammpstrj', 'r') as file:                                         
    lines = file.readlines()                                         
    for i, line in enumerate(lines):                                         
        if "ITEM: TIMESTEP" in line:                                         
            if frame and current_timestep in timesteps:                                         
                frames.append((current_timestep, frame, num_atoms, box_bounds, atom_types))                                         
            frame = []                                         
            current_timestep = int(lines[i + 1].strip())                                         
        elif "ITEM: NUMBER OF ATOMS" in line:                                         
            num_atoms = int(lines[i + 1].strip())                                         
        elif "ITEM: BOX BOUNDS" in line:                                         
            box_bounds = "\n".join([lines[i + 1].strip(), lines[i + 2].strip(), lines[i + 3].strip()])                                         
        elif "ITEM: ATOMS" in line:                                         
            reading_atoms = True                                         
            atom_types = line.strip()  # Capture the ITEM: ATOMS line                                         
        elif reading_atoms:                                         
            frame.append(line.strip())                                         
            if i + 1 < len(lines) and "ITEM: " in lines[i + 1]:                                         
                reading_atoms = False                                         

# Add the last frame if it matches the criteria                                         
if frame and current_timestep in timesteps:                                         
    frames.append((current_timestep, frame, num_atoms, box_bounds, atom_types))                                         

# Write the selected frames to files                                         
for timestep, frame, num_atoms, box_bounds, atom_types in frames:                                         
    with open(f'extracted-confs/frame-{timestep}.lammpstrj', 'w') as f:                                         
        f.write(f"ITEM: TIMESTEP\n{timestep}\n")                                         
        f.write(f"ITEM: NUMBER OF ATOMS\n{num_atoms}\n")                                         
        f.write(f"ITEM: BOX BOUNDS\n{box_bounds}\n")                                         
        f.write(f"{atom_types}\n")                                         
        f.write("\n".join(frame))    
```

### convert-to-QE-input.py
```python
import re
import os
import numpy as np
import ase.io                      
from ase.calculators.espresso import Espresso

################################
# QE options
################################

# Specify the pseudopotentials for the elements          
pseudopotentials = {                              
    "H": "H_ONCV_PBE-1.2.upf",                             
    "C": "C_ONCV_PBE-1.2.upf",                        
    "N": "N_ONCV_PBE-1.2.upf",                                                        
    "O": "O_ONCV_PBE-1.2.upf"                                                                                                  
}                                                                                                                                                                                                                        

# INPUT PARAMETERS                                                       
kpoints = "gamma"                        
offset = (0, 0, 0)                                                                      
ecut = int(os.getenv("ecut"))                                

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
        "ecutwfc": ecut,                                                                 
        "input_dft": "PBE",                          
        "vdw_corr": "grimme-d3"                                   
    },                                                      
    "electrons": {                                                                              
        "mixing_beta": 0.5,                                   
        "electron_maxstep": 1000                             
    }                                                        
}                                                              

# Atom ID to element mapping                                                    
id_to_element = {                                                                   
    1: "H",                                                             
    2: "C",                                    
    3: "N",
    4: "O"                                                          
}                                                                        

seed_number = os.getenv("random_seed")    

# Convert the extracted frames to Quantum ESPRESSO input files       
for filename in os.listdir('extracted-confs'):            
    if filename.endswith('.lammpstrj'):             
        filepath = os.path.join('extracted-confs', filename)
        timestep = filename.split('-')[1].split('.')[0]                           

        # Read atoms from the file         
        atoms = ase.io.read(filepath, format='lammps-dump-text')

        # Map atom IDs to elements       
        atom_ids = atoms.get_atomic_numbers()                                                          
        atom_symbols = [id_to_element[atom_id] for atom_id in atom_ids]                
        atoms.set_chemical_symbols(atom_symbols)                                

        # Write Quantum ESPRESSO input file                                                     
        ase.io.write(f'extracted-confs/pw-RS-{seed_number}-step-{timestep}.in', atoms, format='espresso-in',
                     input_data=input_qe, pseudopotentials=pseudopotentials, kpoints=kpoints, offset=offset)
```
