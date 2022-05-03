# disperseNN

## Install Requirements
`conda create --name dispersenn python=3.8`

`pip install -r requirements.txt` should cover you

## Overview
`disperseNN` has two modes: 
1. prediction

        By specifying the `--predict` flag.

        Input types:
            - VCF
            - tree sequences
            - pre-processed tensors
2. training

        Specify using the `--train` flag.

        Input types:  
      	    - tree sequences
            - pre-processed tensors

Within each mode, you may specify different types of input data, each requiring different sets of command line parameters. 

## Brief instructios with example commands
Below are example commands for each of the different input types.

### e.g. prediction: VCF

`python disperseNN.py --predict --empirical ExampleVCFs/halibut --max_n 100 --num_snps 5000 --phase 1 --polarize 2 --load_weights Saved_models/out136_2400.12_model.hdf5 --training_mean -0.9874806682910889 --training_sd 1.8579295139087375 --num_pred 1000 --out out1 --seed 123451`

### e.g. prediction: tree sequences
`python disperseNN.py --predict --empirical ExampleVCFs/halibut --min_n 50 --max_n 50 --num_snps 5000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --on_the_fly 50 --tree_list tree_list.txt --target_list target_list.txt --width_list width_list.txt --load_weights Saved_models/out136_2400.12_model.hdf5 --training_mean XX --training_sd XX --num_pred 1000 --batch_size 10 --threads 10 --out out1 --seed 123451`

### e.g. prediction: pre-processed tensors

### e.g. training: tree sequences

### e.g. training: pre-processed tensors

## Simulation
The authors of `disperseNN` used the slim recipe _____ to generate training data (tree sequences).

Simulation programs other than SLiM may be used to make training data, as long as the output is processed into tensors of the necessary shape. 