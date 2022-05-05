# disperseNN

## Install requirements
```
(optional: ) conda create --name dispersenn python=3.8

pip install -r requirements.txt 
``` 
1;95;0c

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

Simulation programs other than SLiM may be used to make training data, as long as the output is processed into tensors of the necessary shape. 






## Vignette: example workflow

### Custom simulations
We will analyze a theoretical population of *Internecivus raptus*. Let's assume we have independent estimates from previously studies for the size of the species range and the population density: these values are 50x50 km^2, and 4 individuals per square km, respectively. With values for these nuisance parameters in hand we can design custom training simulations for analyzing &#963;. Furthermore, our *a prior* expectation for the dispersal rate in this species is somewhere between 0.2 and 1.5 km/generation; we want to explore potential dispersal rates in this range.

Let's jump into a new working directory and run the simulations:
```
mkdir -p Temp_wd
cd Temp_wd
mkdir TreeSeqs Targets
n=100
for i in {1..100}
do
    sigma=$(python -c 'import numpy as np; print(np.random.uniform(0.2,1.5))')
    echo "slim -d SEED=$i -d sigma=$sigma -d K=4 -d mu=0 -d r=1e-8 -d W=50 -d G=1e8 -d maxgens=100 -d OUTNAME=\"'TreeSeqs/output'\" ../SLiM_recipes/map12.slim" >> sim_commands.txt
    echo TreeSeqs/output_$i.trees >> tree_list.txt
    echo $sigma > Targets/output_$i.target
    echo Targets/output_$i.target >> target_list.txt
done
parallel -j 2 < sim_commands.txt
```
Note: the carrying capacity in this model, K, corresponds roughly to density. However, to be more precise it would be good to closely document the census size for varying Ks, in order to find the best K to get exactly 4 individuals per square km on average (the census size will fluctuate a bit). 





### Training
Before proceeding, we will separate the sims into two groups: (i) training data and (ii) test data. The latter portion will be held out for testing, later. 
```
head -50 tree_list.txt > training_trees.txt
tail -50 tree_list.txt > test_trees.txt
head -50 target_list.txt > training_targets.txt
tail -50 target_list.txt > test_targets.txt
```

The training step is computationally intensive and should ideally be run on a computing cluster. The `threads` flag can be altered to use more CPUs for processing tree sequences.

Our training command will use a similar settings to the above example "Training: tree sequences as input". In particular, we still need to recapitate the fresh tree sequences, so the `recapitate` flag will be set to True. The min and max *n* are both set to 14 because we want to analyze dispersal in a subset of exactly 14 individuals from our empirical data (see below). We will sample 10x from each from each tree sequence for a total training set of size 1000- this is specified via the `on-the-fly` flag.
```
python ../disperseNN.py --train --min_n 14 --max_n 14 --num_snps 1000 --genome_length 100000000 --recapitate True --mutate True --phase 1 --polarize 2 --tree_list training_trees.txt --target_list training_targets.txt --map_width 50 --edge_width 1.5 --sampling_width 1 --on_the_fly 10 --batch_size 20 --threads 2 --max_epochs 10 --validation_split 0.2 --out out1 --seed 12345 --gpu_num -1
```
Note: we chose to sample away from the habitat edges by 1.5km. This is because the simulation model we artifically reduces fitness near the edges.

(*currently running into a bug on sesame if using GPU. Will come back to it)





### Testing
Next, we will validate the trained model using the held-out test data. This command will use a similar set of flags to the above example "Prediction: tree sequences as input".
```
python ../disperseNN.py --predict --min_n 14 --max_n 14 --num_snps 1000 --genome_length 100000000 --recapitate True --mutate True --phase 1 --polarize 2 --tree_list test_trees.txt --target_list test_targets.txt --map_width 50 --edge_width 1.5 --sampling_width 1 --load_weights out1_model.hdf5 --training_targets training_targets.txt --num_pred 50 --batch_size 2 --threads 2 --max_epochs 10 --out out3 --seed 12345 --gpu_num -1 > val_results.txt
```
Note: here we handed `disperseNN` a list of paths to the targets from training; it re-calculates the mean and standard deviation from training, which it uses to back-transform the new predictions.

This `val_results.txt file` shows that our &#963; estimates are accurate.


    TreeSeqs/output_92.trees -0.0028026921 0.1408058872
    TreeSeqs/output_93.trees -0.3473263223 -0.2757609494
    TreeSeqs/output_94.trees -1.6028874848 -1.863671704
    TreeSeqs/output_95.trees 0.1751299976 -0.0791372587
    TreeSeqs/output_96.trees -0.5800663022 -0.2791748059
    TreeSeqs/output_97.trees 0.3666148056 0.3081696193
    TreeSeqs/output_98.trees -0.5374309538 -0.4900743591
    TreeSeqs/output_99.trees -0.0850628176 -0.0756090524
    TreeSeqs/output_100.trees -1.3386295757 -1.4092516501
    RMSE: 0.22454942093462088





### VCF prep.
Now it's time to prepare our empirical VCF for inference with `disperseNN`. This means taking a subset of individuals that we want to analyze, and other basic filters, e.g. removing indels and non-variants sites. Separately, we want a .locs file with the same prefix as the .vcf.

In our case we want to take a subset of individuals from a particular geographic region, the Scotian Shelf region. Further, we want to include only a single individual per sampling location; this is important because individuals did not have overlapping locations in the training simulations, which might trip up the neural network. Last, because we have the option, let's include only female samples. Below are some example commands that might be used to parse the metadata, but these steps will certainly be different for other empirical tables.

```
# [these commands are gross; but I want to eventually switch over to simulated data, so these steps will change]
cat ../Examples/VCFs/iraptus_meta_full.txt | grep "Scotian Shelf - East" | cut -f 4,5 | sort | uniq > templocs
count=$(wc -l templocs | awk '{print $1}')
for i in $(seq 1 $count); do locs=$(head -$i templocs | tail -1); lat=$(echo $locs | awk '{print $1}'); long=$(echo $locs | awk '{print $2}'); grep $lat ../Examples/VCFs/iraptus_meta_full.txt | awk -v coord=$long '$5 == coord' | awk '$2 == "F"' | shuf | head -1; done > iraptus_meta.txt
cat iraptus_meta.txt  | sed s/"\t"/,/g > iraptus.csv
```

We provide a simple python script for subsetting a VCF for a particular set of individuals, which also filters indels and non-variant sites.
```
python ../Empirical/subset_vcf.py ../Examples/VCFs/iraptus_full.vcf.gz iraptus.csv iraptus.vcf 0 1
```


Last, the sample order in the .locs file should correspond to the sample order in the VCF:
```
count=$(zcat iraptus.vcf.gz | grep -v "##" | grep "#" | wc -w)
for i in $(seq 10 $count); do id=$(zcat iraptus.vcf.gz | grep -v "##" | grep "#" | cut -f $i); grep -w $id iraptus.csv; done | cut -d "," -f 4,5 | sed s/","/"\t"/g > iraptus.locs
```






### Empirical inference
Finally, our command for predicting &#963; from the subsetted VCF:
```

```












## References:

Battey CJ, Ralph PL, Kern AD. Space is the place: effects of continuous spatial structure on analysis of population genetic data. Genetics. 2020 May 1;215(1):193-214.



