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
Before handing an empirical VCF to `disperseNN`, it should undergo basic filtering steps to remove non-variant sites and indels; rare variants should be left in. Furthermore, the VCF should include only the individuals that you intend to analyze. Any number of SNPs can be left in the VCF, because `disperseNN` will draw a random subset. As a final rule for the VCF, only one chromosome may be analyzed at a time; if chromosomes need to be combined to obtain enough SNPs, e.g. RADseq data, change the CHROM and POS columns to represent a single pseudo chromosome with continuous positions. Last, a .locs file should be prepared with two columns corresponding to the lat. and long. spatial coordinates for each indvidual. The order of samples in the .vcf and .locs should match.

Below is an example command for estimating &#963; from a VCF file using a pre-trained model:
```
python disperseNN.py --predict --empirical Examples/VCFs/halibut --max_n 100 --num_snps 5000 --phase 1 --polarize 2 --load_weights Saved_models/out136_2400.12_model.hdf5 --training_mean -0.9874806682910889 --training_sd 1.8579295139087375 --num_pred 10 --out out1 --seed 12345
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
python disperseNN.py --predict --min_n 100 --max_n 100 --num_snps 5000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --tree_list Examples/TreeSeqs/tree_list1.txt --target_list Examples/Targets/target_list1.txt --map_width 50 --edge_width 3 --sampling_width 1  --load_weights Saved_models/out136_2400.12_model.hdf5 --training_mean -0.9874806682910889 --training_sd 1.8579295139087375 --num_pred 1 --batch_size 1 --threads 1 --out out1 --seed 12345
```

New flags, here:
- `min_n`: here, specify both min_n and max_n, to draw a random sample size within that range (or set them equal to each other)
- `genome_length`: this is used for rescaling the genomic positions
- `recapitate`: recapitate the tree sequence 
- `mutate`: add mutations to the tree sequence until the specified number of SNPs are obtained
- `tree_list`: list of paths to the tree sequences
- `target_list`: list of paths to the targets; the order should correspond to the tree list
- `map_width`: width of the training habitat. (Alternatively `width_list` can be used to provide a list of different map widths; the order should correspond to the tree list.)
- `edge_width`: this is the width of edge to 'crop' from the sides of the map. In other words, individuals are sampled edge_width distance from the sides of the map.
- `sampling_width`: value in range (0,1), in proportion to the map width
- `batch_size`: for the data generator
- `threads`: number of threads 






### Prediction: using pre-processed tensors
In some cases we may not want to work with tree sequences, e.g. if the tree sequences are very large or if using a different simulator. Instead it may be useful to pre-process a number of simulations up front (outside of `disperseNN`), and provide the ready-to-go tensors straight to `disperseNN`. Genotypes, genomic positions, sample locations, and the sampling width, should be saved as .npy.

```
python disperseNN.py --predict --preprocess --min_n 100 --max_n 100 --num_snps 5000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --geno_list Examples/Genos/genos_list2.txt --pos_list Examples/Positions/pos_list2.txt --loc_list Examples/Locs/loc_list2.txt --samplewidth_list Examples/SampleWidths/samplewidth_list2.txt --target_list Examples/Targets/target_list2.txt --map_width 50 --edge_width 3 --sampling_width 1  --load_weights Saved_models/out136_2400.12_model.hdf5 --training_mean -0.9874806682910889 --training_sd 1.8579295139087375 --num_pred 1 --batch_size 1 --threads 1 --out out1 --seed 12345
```

- `preprocess`: this flag is used to specify that you're providing pre-processed input tensors
- `geno_list`: list of paths to the genotype tensors (.npy)
- `loc_list`: list of paths to the locations tensors (.npy)
- `pos_list`: list of paths to the positions tensors (.npy)
- `samplewidth_list`: list of paths to the sample width tensors (.npy)






### Training: tree sequences as input
Below is an example command for the training step. This example uses tree sequences as input.
```
python disperseNN.py --train --min_n 10 --max_n 10 --num_snps 1000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --tree_list Examples/TreeSeqs/tree_list1.txt --target_list Examples/Targets/target_list1.txt --map_width 50 --edge_width 3 --sampling_width 1 --on_the_fly 100 --batch_size 10 --threads 1 --max_epochs 100 --validation_split 0.5 --out out1 --seed 12345
```
- `max_epochs`: for training
- `validation_split`: proportion of training datasets to hold out for validation; that is, within-training validation.
- `on_the_fly`: on-the-fly mode takes more than one sample from each tree sequence, augmenting the training set while saving simulation time.






### Training: with pre-processed tensors
As before, pre-processed tensors may be used instead of tree sequences:
```
python disperseNN.py --train --preprocess --min_n 100 --max_n 100 --num_snps 5000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --geno_list Examples/Genos/genos_list2.txt --pos_list Examples/Positions/pos_list2.txt --loc_list Examples/Locs/loc_list2.txt --samplewidth_list Examples/SampleWidths/samplewidth_list2.txt --target_list Examples/Targets/target_list2.txt --batch_size 5 --threads 1 --max_epochs 100 --validation_split 0.5 --out out2 --seed 12345
```
This command used a new combination of flags, but the individual flags should have been described above.








### Simulation
We use the SLiM recipe `SLiM_recipes/map12.slim` to generate training data (tree sequences). The model is borrowed directly from Battey et al. 2021. Certain model parameters are specified on the command line using this recipe. As a demonstration, see the below example command:

```
slim -d SEED=12345 -d sigma=0.2 -d K=5 -d mu=0 -d r=1e-8 -d W=25 -d G=1e8 -d maxgens=100000 -d OUTNAME="'output'" SLiM_recipes/map12.slim
       # Note the two sets of quotes around the output name
```

Simulation programs other than SLiM may be used to make tral mnining data, as long as the output is processed into tensors of the necessary shape. 






## Vignette: example workflow

### Custom simulations
[TODO:
- realistic map? maybe if you have time
- independently-derived values for density (small ,for this example)
- example commands
]

Let's analyze a theoretical population of *Internecivus raptus*. Let's assume we have independent estimates from previously studies for the size of the species range and the population density: these values are 50x50 km^2, and 4 individuals per square km. With values for these nuisance parameters in hand we can design custom training simulations for analyzing &#963;. Furthermore, our *a prior* expectation for the dispersal rate in this species is somewhere between 0.2 and 1.5 km/generation; we want to explore potential dispersal rates in this range.

```
mkdir Temp_wd
cd Temp_wd
```
Note: the carrying capacity in this model, K, corresponds roughly to density. However, to be more precise, it would be good to closely document the census size for varying Ks, in order to find the best K to get exactly 4 individuals per square km (on average; because the census size will fluctuate a bit). 


### Training
- organize slim output
- hold out some data for testing

### Testing

### VCF prep.

### Empirical inference






References
