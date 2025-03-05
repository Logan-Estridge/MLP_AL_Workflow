# Full Script: training.sh
```bash
#!/bin/bash                                                                                                                                                                                                                                                                                                                 
set_plan() {
  export JSON=training.json
  touch ${TRAIN_d}/training.log
  export deepmd_log="${TRAIN_d}/training.log"
}

copy_raw_files() {
  validation_ml_data="${LABEL_VALIDATION_d}/ml_data"
  training_ml_data="${LABEL_TRAINING_d}/ml_data"

  validation_raw_dir="${TRAIN_d}/${polymorph}/raw_files/iter_${iteration}/validation"
  training_raw_dir="${TRAIN_d}/${polymorph}/raw_files/iter_${iteration}/training"

  mkdir -p $training_raw_dir
  mkdir -p $validation_raw_dir

  ln ${validation_ml_data}/*.raw $validation_raw_dir
  mv ${validation_ml_data}/set* $validation_raw_dir
  ln ${training_ml_data}/*.raw $training_raw_dir
  mv ${training_ml_data}/set* $training_raw_dir
}

define_systems() {
  validation_systems=""
  training_systems=""
  for i in $(seq 0 ${iteration}); do
    validation_systems+="../../raw_files/iter_${i}/validation\n"
    training_systems+="../../raw_files/iter_${i}/training\n"
  done

  validation_systems=$(echo -e ${validation_systems})
  training_systems=$(echo -e ${training_systems})

  export validation_systems_json=$(jq -R -s -c 'split("\n") | map(select(length > 0))' <<<"${validation_systems}")
  export training_systems_json=$(jq -R -s -c 'split("\n") | map(select(length > 0))' <<<"${training_systems}")
}

ensemble_submit() {
  slurm_ids=()
  log_files=()
  finished_jobs=()
  cancelled_jobs=()

  date
  echo "Training for polymorph ${polymorph}; iteration ${iteration} started. $(date)" >> ${deepmd_log}
  echo "Training for polymorph ${polymorph}; iteration ${iteration} started."

  for model_id in $(seq 4); do
    export model_id
    sh ${TRAIN_d}/utils/create_training_json.sh

    slurm_id=$(sbatch ${TRAIN_d}/utils/submit-training-GPU.sh | awk '{print $4}')
    echo "training model ${model_id}...; slurm id: ${slurm_id}" >> ${deepmd_log}
    echo "training model ${model_id}...; slurm id: ${slurm_id}"
    slurm_ids+=(${slurm_id})
    log_files+=("_slurm-${slurm_id}.out")
  done
}

wait_jobs_done() {
  while : ; do
    all_done=false
    for slurm_id in "${slurm_ids[@]}"; do
      state=$(sacct -j $slurm_id --format=State%20 -P | grep -E 'COMPLETED|CANCELLED' | awk 'NR < 2 {print $1}')
      if [ "$state" == "COMPLETED" ]; then
        if [[ ! " ${finished_jobs[@]} " =~ " ${slurm_id} " ]]; then
          echo "Model with slurm id ${slurm_id} has finished training." >> ${deepmd_log}
          echo "Model with slurm id ${slurm_id} has finished training."
          finished_jobs+=($slurm_id)
        fi
      elif [ "$state" == "CANCELLED" ]; then
        if [[ ! " ${cancelled_jobs[@]} " =~ " ${slurm_id} " ]]; then
          echo "Model with slurm id ${slurm_id} has been cancelled." >> ${deepmd_log}
          echo "Model with slurm id ${slurm_id} has been cancelled."
          cancelled_jobs+=(${slurm_id})
        fi
      fi
    done

    if [ $(( ${#finished_jobs[@]} + ${#cancelled_jobs[@]} )) -eq 4 ]; then
      all_done=true
    fi

    if [ ${all_done} = true ]; then
      break
    fi

    sleep 10
  done
}

wrap_up() {
  date
  echo "Training for polymorph ${polymorph}; iteration ${iteration} has completed. $(date)" >> ${deepmd_log}
  echo "Training for polymorph ${polymorph}; iteration ${iteration} has completed."

  mv $deepmd_log ${TRAIN_d}/${polymorph}/iter_${iteration}
  for log_file in "${log_files[@]}"; do
    if [ -f ${log_file} ]; then
      mv $log_file ${TRAIN_d}/${polymorph}/iter_${iteration}
    else
      echo "Warning: ${log_file} not found. Skipping..."
    fi
  done
}

submit_jobs() {
  set_plan
  copy_raw_files
  define_systems
  ensemble_submit
  wait_jobs_done
  wrap_up
}                                                                                                                                                                                                                                                                                                                           
submit_jobs 
```

## Outline
```bash
submit_jobs() {
  set_plan
  copy_raw_files
  define_systems
  ensemble_submit
  wait_jobs_done
  wrap_up
} 

submit_jobs
```
- `set_plan` defines some useful variables
- `copy_raw_files` either links or moves the raw data from `ml_data/` to the training directory for the current iteration
- `define_systems` is meant to capture the relative file paths of all of the raw data 
- `ensemble_submit` creates the ensemble of 4 DeePMD-kit input files, submits them and captures the slurm ids
- `wait_jobs_done` ensures all the training slurm jobs finish before moving on to `wrap_up`
- `wrap_up` moves some log files to an appropriate location

## Details
`set_plan` and `copy_raw_files` are self-explanatory. As you can see, I used hardlinks (`ln`) or `mv` in `set_plan` (instead of `cp`) to limit the memory cost of the active learning workflow. 

The function `define_systems` captures the relative file paths of all the raw data from iteration `0` up to the current iteration. To be able to insert this text properly into the DeePMD-kit input file (`training.json`) I use the tool `jq`, which is a command-line JSON processor.
```bash
define_systems() {
  validation_systems=""
  training_systems=""
  for i in $(seq 0 ${iteration}); do
    validation_systems+="../../raw_files/iter_${i}/validation\n"
    training_systems+="../../raw_files/iter_${i}/training\n"
  done

  validation_systems=$(echo -e ${validation_systems})
  training_systems=$(echo -e ${training_systems})

  export validation_systems_json=$(jq -R -s -c 'split("\n") | map(select(length > 0))' <<<"${validation_systems}")
  export training_systems_json=$(jq -R -s -c 'split("\n") | map(select(length > 0))' <<<"${training_systems}")
}
```
The variables `validation_systems_json` and `training_systems_json` will be utilized in the next function, which runs the script `create_training_json.sh`:
```bash
ensemble_submit() {
  slurm_ids=()
  log_files=()
  finished_jobs=()
  cancelled_jobs=()

  date
  echo "Training for polymorph ${polymorph}; iteration ${iteration} started. $(date)" >> ${deepmd_log}
  echo "Training for polymorph ${polymorph}; iteration ${iteration} started."

  for model_id in $(seq 4); do
    export model_id
    sh ${TRAIN_d}/utils/create_training_json.sh

    slurm_id=$(sbatch ${TRAIN_d}/utils/submit-training-GPU.sh | awk '{print $4}')                                                           
    echo "training model ${model_id}...; slurm id: ${slurm_id}" >> ${deepmd_log}
    echo "training model ${model_id}...; slurm id: ${slurm_id}"
    slurm_ids+=(${slurm_id})
    log_files+=("_slurm-${slurm_id}.out")
  done
}
```
### create_training_json.sh
```bash
#!/bin/bash

create_directories() {
  WRK_d="${TRAIN_d}/${polymorph}/iter_${iteration}/${model_id}"
  mkdir -p ${WRK_d}
}

define_variables() {
  descriptor_neuron="[25,50,100]"
  fitting_neuron="[240,240,240]"
  rand1=$((model_id * 10 + 1))
  rand2=$((model_id * 10 + 2))
  rand3=$((model_id * 10 + 3))
  sel="[55,54,18,18]"
  steps="400000"
}

perform_substitutions() {
  declare -A vars=(
    ["training_systems"]="$training_systems_json"
    ["validation_systems"]="$validation_systems_json"
    ["descriptor_neuron"]="$descriptor_neuron"
    ["fitting_neuron"]="$fitting_neuron"
    ["rand1"]="$rand1"
    ["rand2"]="$rand2"
    ["rand3"]="$rand3"                                                                                                                                                              
    ["sel"]="$sel"
    ["steps"]="$steps"
  )

  jq_args=()
  for key in "${!vars[@]}"; do
    value=${vars[$key]}
    if [[ "$value" =~ ^[0-9]+$ ]]; then
      jq_args+=(--argjson "$key" "$value")
    else
      jq_args+=(--arg "$key" "$value")
    fi
  done

  json_template=$(<"${TRAIN_d}/utils/${JSON}.template")

  updated_json=$(jq "${jq_args[@]}" '
    .training.training_data.systems = ($training_systems | fromjson) |
    .training.validation_data.systems = ($validation_systems | fromjson) |
    .model.descriptor.sel = ($sel | fromjson) |
    .model.descriptor.seed = ($rand1 | tonumber) |
    .model.descriptor.neuron = ($descriptor_neuron | fromjson) |
    .model.fitting_net.neuron = ($fitting_neuron | fromjson) |
    .model.fitting_net.seed = ($rand2 | tonumber) |
    .training.seed = ($rand3 | tonumber) |
    .training.stop_batch = ($steps | tonumber)' <<< "$json_template")

  echo "$updated_json" > "$WRK_d/$JSON"
}

create_training_json() {
  create_directories
  define_variables
  perform_substitutions
}

create_training_json
```
To summarize this script simply: it defines some training parameters and inserts them into a template `training.json` file using `jq`. The template `training.json.template` file looks like so:
```json
{                                                                                                                                                                                   
    "_comment": "that's all",                                         
    "model": {                                         
      "type_map": ["H", "C", "N", "O"],                                         
        "descriptor": {                                         
            "type": "se_e2_a",                                         
            "sel": [],                                         
            "rcut_smth": 0.5,                                         
            "rcut": 6.0,                                         
            "neuron": [],                                         
            "axis_neuron": 4,                                         
            "seed": "",                                         
            "_comment": " that's all"                                         
        },                                         
        "fitting_net": {                                         
            "neuron": [],                                         
            "resnet_dt": false,                                         
            "seed": "",                                         
            "_comment": " that's all"                                         
        },                                         
        "_comment": " that's all"                                         
    },                                         
    "learning_rate": {                                         
        "start_lr": 0.002,                                         
        "stop_lr": 7.4e-08,                                         
        "decay_steps": 5000,                                         
        "_comment": "that's all"                                         
    },                                         
    "loss": {                                         
        "start_pref_e": 0.02,                                         
        "limit_pref_e": 2,                                         
        "start_pref_f": 1000,                                         
        "limit_pref_f": 1,                                         
        "start_pref_v": 0,                                         
        "limit_pref_v": 0,                                         
        "_comment": " that's all"                                         
    },                                         
    "training": {                                         
        "stop_batch": "",                                         
        "seed": "",                                         
        "_comment": "that's all",                                         
        "disp_file": "lcurve.out",                                         
        "disp_freq": 1000,                                         
        "numb_test": 5,                                         
        "save_freq": 20000,                                         
        "save_ckpt": "model.ckpt",                                         
        "disp_training": true,                                         
        "time_training": true,                                         
        "tensorboard": false,                                         
        "tensorboard_log_dir": "log",                                         
        "profiling": false,                                         
        "profiling_file": "timeline.json",                                         
        "validation_data": {                                         
          "systems": [],                                         
          "batch_size": "auto"                                         
        },                                         
        "training_data": {                                         
          "systems": [],                                         
          "batch_size": "auto"                                         
        }                                         
    }                                         
}   
```
Following the formats of `create_training_json.sh` and `training.json.template` one can use `jq` to substitute other training parameters into the `training.json` files, if desired. For example, the `rcut`, if testing different values for the cutoff. 

### Back to training.sh
The second part of the `for` loop in `ensemble_submit` is responsible for submitting the slurm jobs (and capturing the slurm ids) to train the ensemble of models:
```bash
for model_id in $(seq 4); do
    export model_id
    sh ${TRAIN_d}/utils/create_training_json.sh

    slurm_id=$(sbatch ${TRAIN_d}/utils/submit-training-GPU.sh | awk '{print $4}')                                                           
    echo "training model ${model_id}...; slurm id: ${slurm_id}" >> ${deepmd_log}
    echo "training model ${model_id}...; slurm id: ${slurm_id}"
    slurm_ids+=(${slurm_id})
    log_files+=("_slurm-${slurm_id}.out")
done
```
the script to submit the training jobs looks like follows:
### submit-training-GPU.sh
```bash
#!/bin/bash -l                                         
#SBATCH -e _slurm-%j.out                                         
#SBATCH -o _slurm-%j.out                                         
#SBATCH --cpus-per-task=8       ## Number of cpus used per task                                         
#SBATCH --gpus-per-task=1                                         
#SBATCH --nodes=1                   ## Number of nodes to be used                                         
#SBATCH --ntasks=1                                         
#SBATCH -p valsson_ml.gpu                ## Partition/jobqueue to be used                                         
#SBATCH --gres-flags=enforce-binding                                         
#SBATCH -t 24:00:00                                         

module purge                                         
module load apptainer                                         
export OMP_NUM_THREADS=8                                         
export TF_INTRA_OP_PARALLELISM_THREADS=4                                         
export TF_INTER_OP_PARALLELISM_THREADS=2                                         
export JOB_d="${polymorph}/iter_${iteration}/${model_id}"                                                                                                                           

train_from_scratch() {                                         
  apptainer exec --nv --bind "${TRAIN_d}":/job \                                         
    /storage/nas_scr/shared/groups/valsson/Logan-Runs/containers/deepmd-kit_v2.1.1.sif \                                         
    bash -c "cd /job/${JOB_d}; dp train ${JSON}"                                         
}                                         

train_freeze_and_compress_model() {                                         
  apptainer exec --nv --bind "${TRAIN_d}":/job \                                         
    /storage/nas_scr/shared/groups/valsson/Logan-Runs/containers/deepmd-kit_v2.1.1.sif \                                         
    bash -c "cd /job/${JOB_d}; dp freeze -o graph-${model_id}.pb && dp compress -i graph-${model_id}.pb -o graph-compress-${model_id}.pb"                                         
}                                         

train_one_shot() {                                         
  train_from_scratch                                         
  train_freeze_and_compress_model                                         
}                                         

train_one_shot   
```
The upper half is a simple `slurm`-type submission script, defining the parameters of submission. `JOB_d` defines the location of the ensemble of models relative to `${TRAIN_d}`, as defined in `active_learning.sh`. 

The function `train_from_scratch` uses apptainer to execute the docker image of the DeePMD-kit I located to `/storage/nas_scr/shared/groups/valsson/Logan-Runs/containers/deepmd-kit_v2.1.1.sif`. I also "`bind`" the `${TRAIN_d}` directory to "`/job`". This allows the docker container, and thus the DeePMD-kit, to see any files located in `${TRAIN_d}`, which is needed as the `training.json` file is referencing the raw data files via a relative path (which are located within `${TRAIN_d}`). As a caveat, I tried several times to use absolute paths to avoid the mess of relative paths with this method. But this is not possible (unless I bind `/` to `/job`) as any directories outside of `${TRAIN_d}` will be invisible to the docker container. 

Finally, once the ensemble of models are trained, the `train_freeze_and_compress_model` function handles compressing the outputs into `graph-compress-?-.pb` form. 

### Back to training.sh (again)
`wait_jobs_done` was described in detail in [[4 - Journal/PABA MLP/Documentation/labeling|labeling]], and `wrap_up` requires no further explanation. Thus, an ensemble of models can be trained for any arbitrary iteration of active learning. 
