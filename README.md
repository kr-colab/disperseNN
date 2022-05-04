# disperseNN

## Install requirements
```
(optional: ) conda create --name dispersenn python=3.8

pip install -r requirements.txt 
``` 


## Overview
`disperseNN` has two modes: 
1. prediction

        Command line flag:
            --predict

        Input types:
            - VCF
            - tree sequences
            - pre-processed tensors
2. training

        Command line flag:
            --train

        Input types:  
      	    - tree sequences
            - pre-processed tensors

Within each mode- prediction or training- you may specify different types of input data, each requiring its own set of additional command line parameters; details below. 







## Brief instructions with example commands

### Prediction: using a VCF as input
An empirical VCF should undergo some basic filtering steps, e.g. to remove non-variant sites, extra samples, etc. This can be accomplished using the build in script `subset_vcf.py`.

   TODO

Next, you need a locations file with two columns for lat and long. **** oh dang, or is it the other way around?

Below is an example command for estimating &#963; from a VCF file using a pre-trained model:
```
python disperseNN.py --predict --empirical ExampleVCFs/halibut --max_n 100 --num_snps 5000 --phase 1 --polarize 2 --load_weights Saved_models/out136_2400.12_model.hdf5 --training_mean -0.9874806682910889 --training_sd 1.8579295139087375 --num_pred 1000 --out out1 --seed 123451
```

Explanation of command line values:
- `empirical`: this flag is specific to analyzing VCFs. Give it the shared prefix for the .vcf and .locs files (i.e. no '.vcf' or '.locs')
- `max_n`: sample size
- `num_snps`: number of snps to analyze. This number equals num_snps in the loaded model, but is probably fewer than the VCF lines.
- `phase`: '1' for unphased, '2' for phased 
- `polarize`: '1' for polarized, '2' for unpolarized
- `load_weights`: saved model or weights to load
- `training_mean`: mean from the training targets
- `training_sd`: standard deviation from the training targets
- `num_pred`: number of datasets to predict; here, the number of bootstrap replicates
- `out`: output prefix
- `seed`: random number seed








### Prediction: tree sequences as input
If you want to predict &#963; in simulated tree sequences, an example command is:
```
python disperseNN.py --predict --min_n 50 --max_n 50 --num_snps 5000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --tree_list tree_list.txt --target_list target_list.txt --width_list width_list.txt --sampling_width 1  --load_weights Saved_models/out136_2400.12_model.hdf5 --training_mean -0.9874806682910889 --training_sd 1.8579295139087375 --num_pred 1000 --batch_size 10 --threads 10 --out out1 --seed 123451
```

New flags, here:
- `min_n`: here, specify both min_n and max_n, to draw a random sample size within that range (or set them equal to each other)
- `genome_length`: this is used for rescaling the genomic positions
- `recapitate`: recapitate the tree sequence 
- `mutate`: add mutations to the tree sequence until the specified number of SNPs are obtained
- `tree_list`: list of paths to the tree sequences
- `target_list`: list of paths to the targets; the order should correspond to the tree list
- `width_list`: list of map widths; the order should correspond to the tree list. (Alternatively, a fixed map width can be provided with `map_width`.)
- `sampling_width`: value in range (0,1), in proportion to the map width
- `batch_size`: for the data generator
- `threads`: number of threads 






### Prediction: using pre-processed tensors
In some cases we may not want to work with tree sequences, e.g. if the tree sequences are very large or if using a different simulator. Instead it may be useful to pre-process a number of simulations up front (outside of `disperseNN`), and provide the ready-to-go tensors straight to `disperseNN`. Genotypes, genomic positions, sample locations, and the sampling width, should be saved as .npy.

```
python disperseNN.py --predict --min_n 50 --max_n 50 --num_snps 5000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --preprocess --geno_list geno_list.txt --loc_list loc_list.txt --pos_list pos_list.txt --samplewidth_list sample_widths.txt --target_list target_list.txt --load_weights Saved_models/out136_2400.12_model.hdf5 --training_mean -0.9874806682910889 --training_sd 1.8579295139087375 --num_pred 1000 --batch_size 10 --threads 10 --out out1 --seed 123451
```

- `preprocess`: this flag is used to specify that you're providing pre-processed input tensors
- `geno_list`: list of paths to the genotype tensors (.npy)
- `loc_list`: list of paths to the locations tensors (.npy)
- `pos_list`: list of paths to the positions tensors (.npy)
- `samplewidth_list`: list of paths to the sample width tensors (.npy)






### Training: tree sequences as input
Below is an example command for the training step. This example uses tree sequences as input.
```
python disperseNN.py --train --min_n 50 --max_n 50 --num_snps 5000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --tree_list tree_list.txt --target_list target_list.txt --width_list width_list.txt --sampling_width 1 --on-the-fly 50 --batch_size 10 --threads 10 --max_epochs 100 --validation_split 0.2 --out out1 --seed 123451
```
- `max_epochs`: for training
- `validation_split`: proportion of training datasets to hold out for validation; that is, within-training validation.
- `on_the_fly`: on-the-fly mode takes more than one sample from each tree sequence, augmenting the training set while saving simulation time.
- `edge_width`: this is the width of edge to 'crop' from the sides of the map. In other words, individuals are sampled edge_width distance from the sides of the map.






### Training: with pre-processed tensors
As before, pre-processed tensors may be used instead of tree sequences:
```
python disperseNN.py --train --min_n 50 --max_n 50 --num_snps 5000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --geno_list geno_list.txt --loc_list loc_list.txt --pos_list pos_list.txt --samplewidth_list sample_widths.txt --target_list target_list.txt --sampling_width 1 --on-the-fly 50 --batch_size 10 --threads 10 --max_epochs 100 --validation_split 0.2 --out out1 --seed 123451
```
This command used a new combination of flags, but the individual flags should have been described above.








## Simulation
The authors of `disperseNN` used the SLiM recipe _____ to generate training data (tree sequences).

Simulation programs other than SLiM may be used to make training data, as long as the output is processed into tensors of the necessary shape. 
