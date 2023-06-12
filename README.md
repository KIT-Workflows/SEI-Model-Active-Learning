When publishing results obtained with this set of **WaNos**, please consider citing it. [![DOI](https://zenodo.org/badge/440164995.svg)](https://zenodo.org/badge/latestdoi/440164995)

# SEI-Model-Active-Learning
Workflow for Solid Electrolyte Interface (SEI) model within Active Learning approach

## Python dependencies

* pip install torch torchvision torchaudio
* pip install gpytorch
* conda install scipy=1.7
* pip install seaborn=0.12.2
* pip install scikit-learn==1.2.1
* pip install pyyaml

## Starting the Workflow
There are two options for using the workflow: 1) colaborating team installing and using the workflow on their own machines, 2) colaborating team that want to choose the INT-Nano resources.
For the first option, these are the checklist: 
 - Making sure starting multiple jobs is allowed on your cluster (for the advanced loop in the workflow)
 - Making sure you are assigned to a similar computational power and architecture (more import for large clusters)
 - Making sure you are allowed to upload pickle files to the cluster
 
After making sure these conditions are met, the steps for starting the workflow are:
 - Recreating the 50,000 dataset on you own cluster (necessary instructions can be provided by us)
 - Receiving the Simstack server from us, which should be installed on your cluster
 - Downloading the Simstack client, and the WaNOs
 - Starting the workflow

For the second option, the colaborating team needs only to obtain access to the INT-Nano cluster, have the Simstack server installed for them, and download the Simstack client program and the WaNOs.
From now, we assume all the preconditions for starting the workflow are met. 

Step 1) The pickle file (npy) is the collection of 50,000 output from the descriptor that is applied to kMC model output, and the json file is the main 50,000 reaction barrier. These files are stored in a folder and their adresses are passed to the SEI-Init WaNO. 
