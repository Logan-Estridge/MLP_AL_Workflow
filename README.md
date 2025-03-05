# MLP_AL_Workflow
The "automatic" active learning workflow I have developed in bash/python to train a machine learning potential for the polymorphs of p-aminobenzoic acid. 

## Basic Structure / Terminology
There are three main directories in the workflow, corresponding to the three main processes involved in active learning to train a machine learning potential (MLP). These processes are performed iteratively in the following order to train an accurate MLP:
- labeling
- training
- exploration

Labeling is where configurations of the system are assigned energies/forces via DFT calculations (Quantum Espresso package, QE). 

Training is where an ensemble of deep neural networks, aka DeePMD models, are trained using the energy/force data from the labeling step using the DeePMD-kit. The only difference between DeePMD models for a given iteration of active learning is the value of the "random seeds", in the DeePMD-kit input file, which controls the values of the initial weights and biases in the neural networks.

Finally, exploration is where a trained DeePMD model is used as the potential in a molecular dynamics (LAMMPS: https://www.lammps.org/#gsc.tab=0) simulation, to generate new structures (i.e. explore the configuration space). Following the query by committee method, the model deviations between DeePMD models is monitored for high-deviation structures. These structures are then selected to be labeled for the next iteration of active learning. For more information read the following papers:
-  [https://doi.org/10.1063/5.0160326](https://doi.org/10.1063/5.0160326) = great overview of the topic
- https://doi.org/10.1016/j.cpc.2020.107206 = specific info that inspired this active learning workflow
## Codes Required
1. Install gpu-accelerated versions of the DeePMD-kit, and LAMMPS, for training and exploration.
	- see:  
		- https://docs.deepmodeling.com/projects/deepmd/en/stable/install/easy-install.html (for DeePMD cpu / lammps gpu and cpu version)
		- https://catalog.ngc.nvidia.com/orgs/hpc/containers/deepmd-kit (for gpu-accelerated DeePMD)
2. Install Quantum Espresso cpu/gpu version for DFT calculations
	- QE already on cruntch4 (QE/7.1/imkl_gcc12.2_ompi4.1.4)
3. GROMACS for classical MD simulations (generating initial data set)
	- GROMACS already on cruntch4: place the following in `~/.bash_profile`
```bash
GromacsVersion=gromacs-2022.5_plumed-2.8.2                                         
source /cm/shared/apps/VALSSON/gromacs/${GromacsVersion}/load-gromacs-plumed.sh
```
## Generate Initial Dataset
Use GROMACS to perform a classical MD simulation of the polymorph/system of interest. Extract an appropriate number of structures (100-1,000) from the production NPT trajectory in the `*.gro` format. **It is important in this step to set the name of the `*.gro` structures to include a 5 digit string corresponding to the timestep of the extracted frame.** 

## Reading Order
1. README
2. Quick Start
3. Details
4. labeling
5. training
6. exploration
