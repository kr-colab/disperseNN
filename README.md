# disperseNN

## Install requirements
```
pip install -r requirements.txt 
``` 

You will also need to install SLiM v3.7, in order to run sims and to follow along in the below Vignette. Instructions here: https://messerlab.org/slim/



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

Below is an example command for estimating &#963; from a VCF file using a pre-trained model (should take <30s to run):
```
python disperseNN.py --predict --empirical Examples/VCFs/halibut --max_n 100 --num_snps 5000 --phase 1 --polarize 2 --load_weights Saved_models/out136_2400.12_model.hdf5 --training_mean -0.9874806682910889 --training_sd 1.8579295139087375 --num_pred 10 --out out1 --seed 12345 --gpu_number -1
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
- `gpu_number` : a GPU index. To avoid using available GPUs, specify -1. (default=-1)







### Prediction: tree sequences as input
If you want to predict &#963; in simulated tree sequences, an example command is (should take <30s to run):
```
python disperseNN.py --predict --min_n 100 --max_n 100 --num_snps 5000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --tree_list Examples/TreeSeqs/tree_list1.txt --target_list Examples/Targets/target_list1.txt --map_width 50 --edge_width 3 --sampling_width 1  --load_weights Saved_models/out136_2400.12_model.hdf5 --training_mean -0.9874806682910889 --training_sd 1.8579295139087375 --num_pred 1 --batch_size 1 --threads 1 --out out1 --seed 12345 --gpu_number -1
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







### Training: tree sequences as input
Below is an example command for the training step. This example uses tree sequences as input (runs for minutes to hours, depending on threads).
```
python disperseNN.py --train --min_n 10 --max_n 10 --num_snps 1000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --tree_list Examples/TreeSeqs/tree_list1.txt --target_list Examples/Targets/target_list1.txt --map_width 50 --edge_width 3 --sampling_width 1 --on_the_fly 100 --batch_size 10 --threads 1 --max_epochs 10 --validation_split 0.5 --out out1 --seed 12345 --gpu_number -1
```
- `max_epochs`: for training
- `validation_split`: proportion of training datasets to hold out for validation; that is, within-training validation.
- `on_the_fly`: on-the-fly mode takes more than one sample from each tree sequence, augmenting the training set while saving simulation time.











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

Let's first jump into a new working directory:
```
mkdir -p Temp_wd
cd Temp_wd
```

Next, install SLiM locally: 
```
wget https://github.com/MesserLab/SLiM/releases/download/v3.7.1/SLiM.zip
unzip SLiM.zip
mkdir build
cd build
cmake ../SLiM
make slim
cd ../
```

Now we can run the simulations (runs for a few minutes, to an hour, depending on threads)::
```
mkdir TreeSeqs Targets
n=100
for i in {1..100}
do
    sigma=$(python -c 'import numpy as np; print(np.random.uniform(0.2,1.5))')
    echo "build/slim -d SEED=$i -d sigma=$sigma -d K=4 -d mu=0 -d r=1e-8 -d W=50 -d G=1e8 -d maxgens=100 -d OUTNAME=\"'TreeSeqs/output'\" ../SLiM_recipes/map12.slim" >> sim_commands.txt
    echo $sigma > Targets/output_$i.target
    echo Targets/output_$i.target >> target_list.txt
done
parallel -j 2 < sim_commands.txt
```
Note: the carrying capacity in this model, K, corresponds roughly to density. However, to be more precise it would be good to closely document the census size for varying Ks, in order to find the best K to get exactly 4 individuals per square km on average (the census size will fluctuate a bit). 


One more step before training: we need to recapitate the tree sequences. Although `disperseNN` has an option to recapitate during training, it'll save us time in the long run if we recapitate up front (runs for a few minutes, to an hour, depending on threads):
```
for i in {1..100};
do
	echo "python -c 'import pyslim; ts=pyslim.load(\"TreeSeqs/output_$i.trees\"); Ne=len(ts.individuals_alive_at(0)); ts=pyslim.recapitate(ts,recombination_rate=1e-8,ancestral_Ne=Ne,random_seed=$i); ts.dump(\"TreeSeqs/output_$i"_"recap.trees\")'" >> recap_commands.txt
	echo TreeSeqs/output_$i"_"recap.trees >> tree_list.txt
done   
parallel -j 2 < recap_commands.txt
```




### Training
Before proceeding, we will separate the sims into two groups: (i) training data and (ii) test data. The latter portion will be held out for testing, later. 
```
head -50 tree_list.txt > training_trees.txt
tail -50 tree_list.txt > test_trees.txt
head -50 target_list.txt > training_targets.txt
tail -50 target_list.txt > test_targets.txt
```

The training step is computationally intensive and should ideally be run on a computing cluster. The `threads` flag can be altered to use more CPUs for processing tree sequences. Using 20 dedicated threads (and batch size=20), this step should take several hours; however, if only 1-2 threads are used, the training step will take days.

Our training command will use a similar settings to the above example "Training: tree sequences as input". Of note, the min and max *n* are both set to 14 because we want to analyze dispersal in a subset of exactly 14 individuals from our empirical data (see below). We will sample 100x from each from each tree sequence for a total training set of size 5000- this is specified via the `on-the-fly` flag.
```
python ../disperseNN.py --train --min_n 14 --max_n 14 --num_snps 1000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --tree_list training_trees.txt --target_list training_targets.txt --map_width 50 --edge_width 1.5 --sampling_width 1 --on_the_fly 100 --batch_size 20 --threads 2 --max_epochs 5 --validation_split 0.2 --out out1 --seed 12345 --gpu_number -1
```
Note: we chose to sample away from the habitat edges by 1.5km. This is because the simulation model we artifically reduces fitness near the edges.







### Testing
Next, we will validate the trained model using the held-out test data. This command will use a similar set of flags to the above example "Prediction: tree sequences as input" (should take a minute or two to run).
```
python ../disperseNN.py --predict --min_n 14 --max_n 14 --num_snps 1000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --tree_list test_trees.txt --target_list test_targets.txt --map_width 50 --edge_width 1.5 --sampling_width 1 --load_weights out1_model.hdf5 --training_targets training_targets.txt --num_pred 50 --batch_size 2 --threads 2 --out out2 --seed 12345 --gpu_number -1 > val_results.txt
```
Note: here we handed `disperseNN` a list of paths to the targets from training; it re-calculates the mean and standard deviation from training, which it uses to back-transform the new predictions.

This `val_results.txt` file shows that our estimates are accurate, therefore `disperseNN` was successful at learning to estimate &#963;.
```
TreeSeqs/output_93_recap.trees 0.7065747195 1.0331190445
TreeSeqs/output_94_recap.trees 0.2013143857 0.2030430197
TreeSeqs/output_95_recap.trees 1.1914010858 1.1188061793
TreeSeqs/output_96_recap.trees 0.5598612453 0.7142859062
TreeSeqs/output_97_recap.trees 1.4428420374 0.7139526873
TreeSeqs/output_98_recap.trees 0.5842472843 0.593893058
TreeSeqs/output_99_recap.trees 0.9184545874 0.6911014651
TreeSeqs/output_100_recap.trees 0.2622047542 0.2629951186
RMSLE: 0.3172221601286179
RMSE: 0.29222109958735537
```








### VCF prep.
Now it's time to prepare our empirical VCF for inference with `disperseNN`. This means taking a subset of individuals that we want to analyze, and other basic filters, e.g. removing indels and non-variants sites. Separately, we want a .locs file with the same prefix as the .vcf.

In our case we want to take a subset of individuals from a particular geographic region, the Scotian Shelf region. Further, we want to include only a single individual per sampling location; this is important because individuals did not have overlapping locations in the training simulations, which might trip up the neural network. Below are some example commands that might be used to parse the metadata, but these steps will certainly be different for other empirical tables.

```
# [these commands are gross; but I want to eventually switch over to simulated data, so these steps will change]
cat ../Examples/VCFs/iraptus_meta_full.txt | grep "Scotian Shelf - East" | cut -f 4,5 | sort | uniq > templocs
count=$(wc -l templocs | awk '{print $1}')
for i in $(seq 1 $count); do locs=$(head -$i templocs | tail -1); lat=$(echo $locs | awk '{print $1}'); long=$(echo $locs | awk '{print $2}'); grep $lat ../Examples/VCFs/iraptus_meta_full.txt | awk -v coord=$long '$5 == coord' | shuf | head -1; done > iraptus_meta.txt
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
gunzip iraptus.vcf.gz
```






### Empirical inference
Finally, our command for predicting &#963; from the subsetted VCF (should take less than 30s to run):

```
python ../disperseNN.py --predict --empirical iraptus --min_n 14 --max_n 14 --num_snps 1000 --phase 1 --polarize 2 --load_weights out1_model.hdf5 --training_targets training_targets.txt --num_pred 100 --out out2 --seed 12345 --gpu_number -1
```
Note: `num_pred`, here, specifies how many bootstrap replicates to perform, that is how many replicate draws of 1000 SNPs.











## References:

Battey CJ, Ralph PL, Kern AD. Space is the place: effects of continuous spatial structure on analysis of population genetic data. Genetics. 2020 May 1;215(1):193-214.



