 disperseNN

- [disperseNN](#dispersenn)
  - [Install requirements](#install-requirements)
  - [Overview](#overview)
  - [Brief instructions with example commands](#brief-instructions-with-example-commands)
    - [Prediction: using a VCF, sample locations, and a pre-trained model as inputs](#prediction-using-a-vcf-sample-locations-and-a-pre-trained-model-as-inputs)
    - [Prediction: tree sequences as input](#prediction-tree-sequences-as-input)
    - [Training: tree sequences as input](#training-tree-sequences-as-input)
    - [Simulation](#simulation)
  - [Vignette: example workflow](#vignette-example-workflow)
    - [Custom simulations](#custom-simulations)
    - [Training](#training)
    - [Testing](#testing)
    - [VCF prep](#vcf-prep)
    - [Empirical inference](#empirical-inference)
  - [References](#references)

`disperseNN` is a Machine Learning framework to predict &#963;, the expected per generation displacement distance between offspring and their parent(s), from genetic variation data.
`disperseNN` uses training data generated from simulations over a broad range of parameters.
See [Smith et al.](https://kr-colab.github.io/) for a comprehensive description of the method.

## Install requirements

First clone this repository:

```bash
git clone https://github.com/chriscrsmith/disperseNN.git
cd disperseNN/
```

Next create a `conda` environment using our provided `yaml` file. 
This will create a virtual environment, and then install all the
dependencies. Should work with `mamba` too!

```bash
conda env create -f conda_env.yml
```

After `conda` has done its thing, activate the new environment
to use `disperseNN`
```bash
conda activate disperseNN
```

## Overview

`disperseNN` has two modes:

### 1. prediction

  Command line flag:  `--predict`

  Input types:
  * VCF
  * tree sequences
  * pre-processed numpy arrays
  
### 2. training

  Command line flag:  `--train`

  Input types:  
  * tree sequences
  * pre-processed numpy arrays

Within each mode- `predict` or `train`- you may specify different types of input data, each requiring its own set of additional command line parameters; details below.

## Brief instructions with example commands

### Prediction: using a VCF, sample locations, and a pre-trained model as inputs

While `disperseNN` can be trained from scratch, we recommend trying the pre-trained model provided in this repository first.
Before handing an empirical VCF to `disperseNN`, it should undergo basic filtering steps to remove non-variant sites and indels; rare variants should be left in.
Furthermore, the VCF should include only the individuals that you intend to analyze.
At least 5,000 SNPs are required for the current model,
but more than 5,000 SNPs can be left in the VCF because `disperseNN` will draw a random subset.
Last, a .locs file should be prepared with two columns corresponding to the lat. and long. spatial coordinates for each individual.


Before running any of the example commands, we recommend setting up a new working directory:

```bash
mkdir -p temp_wd/
```

Below is an example command for estimating &#963; from a VCF file using a pre-trained model (should take <30s to run).

```bash
python disperseNN.py \
  --predict \
  --load_weights Saved_models/out136_2400.12_model.hdf5 \
  --training_params Saved_models/out136_2400.12_training_params.npy \
  --empirical Examples/VCFs/halibut \
  --num_reps 10 \
  --out temp_wd/out_vcf \
  --seed 12345
```


Explanation of command line values:
- `load_weights`: saved model or weights to load.
The above command points to a particular pre-trained model, `out136_2400.12_model.hdf5`,
but instructions for how to train a new model from scratch are described below.
- `training_params`: a single file with several parameters used to train the above model. Included are: the mean and standard deviation used to normalize the training targets, and the number of SNPs and maximum sample size used to train the above model. In this case, the number of SNPs is 5,000 and the max sample size is 100.
- `empirical`: this flag is specific to analyzing VCFs. Give it the shared prefix for the .vcf and .locs files (i.e. without '.vcf' or '.locs')
This number equals num_snps in the loaded model, but is probably fewer than the VCF lines.
- `num_reps`: number of repeated draws (of 5,000 SNPs in this case) from the VCF, with one prediction per subset.
- `out`: output prefix.
- `seed`: random number seed.

In addition to printing information about the model architecture to standard output, this command will also create a new file called `temp_wd/out_vcf_predictions.txt`, containing:

```bash
Examples/VCFs/halibut_0 5.6412617497
Examples/VCFs/halibut_1 6.037629072
Examples/VCFs/halibut_2 11.5903892079
Examples/VCFs/halibut_3 6.7408159579
Examples/VCFs/halibut_4 5.9518683625
Examples/VCFs/halibut_5 7.9183494346
Examples/VCFs/halibut_6 5.3469995841
Examples/VCFs/halibut_7 6.2734158333
Examples/VCFs/halibut_8 8.9114959234
Examples/VCFs/halibut_9 6.2295782895
```

Where each line is one of the 10 predictions of &#963; using a random subset of 5K SNPs.

### Prediction: tree sequences as input

If you want to predict &#963; in simulated tree sequences, such as those generated by `msprime` and `SLiM`, an example command is (should take <30s to run):

First make a file listing the paths to the tree sequences:

```bash
ls Examples/TreeSeqs/*trees > temp_wd/tree_list1.txt
```

```bash
python disperseNN.py \
  --predict \
  --load_weights Saved_models/out136_2400.12_model.hdf5 \
  --training_params Saved_models/out136_2400.12_training_params.npy \
  --tree_list temp_wd/tree_list1.txt \
  --recapitate False \
  --mutate True \
  --min_n 90 \
  --edge_width 3 \
  --sampling_width 1  \
  --batch_size 1 \
  --num_pred 3 \
  --out temp_wd/out_treeseq \
  --seed 12345 \
```

In addition to the flags already introduced in the VCF example, the additional flags for this command are:
- `tree_list`: list of paths to the analyzed tree sequences. &#963; values and map widths are extracted directly from the tree sequence.
- `recapitate`: recapitate the tree sequence. Here, we have specified 'False' because the provided tree sequences are already recapitated.
- `mutate`: add mutations to the tree sequence until the specified number of SNPs are obtained (5,000 in this case).
- `min_n`: specifies the minimum sample size. Recall that `out136_2400.12_training_params.npy` specifies a maximum sample size of 100, thus a random sample size between 90 and 100 will be drawn in this case. The model `out136_2400.12_model.hdf5` requires sample size between 10 and 100.
- `edge_width`: this is the width of edge to 'crop' from the sides of the map. In other words, individuals are sampled edge_width distance from the sides of the map.
- `sampling_width`: value in range (0,1), in proportion to the map width.
- `batch_size`: for the data generator.
- `num_pred`: this flag specifies how many simulations from the `tree_list` to predict with. 
   Note: `--num_pred` is distinct from `--num_reps`; if you were to add the `--num_reps` 
   flag to this command, the output would include repeated draws of 5,000 SNPs from each sample of `n` individuals. 


Similar to the previous example, this will generate a file called `temp_wd/out_treeseq_predictions.txt` containing:

```bash
Examples/TreeSeqs/output_sigma0.2to3_K5_W50_100gens_98180132_Ne12336_recap.trees 0.4588403008 16.4714975747
Examples/TreeSeqs/output_sigma0.2to3_K5_W50_100gens_98217910_Ne11232_recap.trees 1.8739656258 3.51678821
Examples/TreeSeqs/output_sigma0.2to3_K5_W50_100gens_99284440_Ne11375_recap.trees 2.0513000433 3.6275611008
```

Where the second and third columns contain the true and predicted &#963; for each simulation.

### Training: tree sequences as input

Below is an example command for the training step.
This example uses tree sequences as input (runs for minutes to hours, depending on threads).

```bash
python disperseNN.py \
  --train \
  --tree_list temp_wd/tree_list1.txt \
  --recapitate False \
  --mutate True \
  --min_n 10 \
  --max_n 10 \
  --num_snps 1000 \
  --edge_width 3 \
  --sampling_width 1 \
  --num_samples 100 \
  --batch_size 10 \
  --max_epochs 10 \
  --out temp_wd/out1 \
  --seed 12345 \
  --threads 1 \
```

- `max_n`: paired with `min_n` to describe the range of sample sizes to drawn from. Set `min_n` equal to `max_n` to use a fixed sample size.
- `num_snps`: the number of SNPs to use as input for the CNN.
- `max_epochs`: for training
- `num_samples`: this is the number of repeated draws of `n` individuals to take from each tree sequence. Note: this is different than `num_reps` and `num_pred`.
- `threads`: number of threads to use for multiprocessing.






### Simulation

In some cases, the pre-trained model provided may not be appropriate for your data.
In this case, it is possible to train new model from scratch from new a simulated training set.
We use the SLiM recipe `SLiM_recipes/bat20.slim` to generate training data (tree sequences).
The model is adapted from [Battey et al. (2020)](https://doi.org/10.1534/genetics.120.303143),
but certain model parameters are specified on the command line.

As a demonstration, see the below example command:

```bash
slim -d SEED=12345 \
     -d sigma=0.2 \
     -d K=5 \
     -d mu=0 \
     -d r=1e-8 \
     -d W=50 \
     -d G=1e8 \
     -d maxgens=100000 \
     -d OUTNAME="'temp_wd/output'" \
     SLiM_recipes/bat20.slim
       # Note the two sets of quotes around the output name
```

Command line arguments are passed to SLiM using the `-d` flag followed the the variable name as it appears in the recipe file.

- `SEED` - a random seed to reproduce the simulation results.
- `sigma` - the dispersal parameter.
- `K` - carrying capacity.
- `mu` - per base per genertation mutation rate.
- `r` -  per base per genertation recombination rate.
- `W` - the height and width of the geographic spatial boundaries.
- `G` - total size of the simulated genome.
- `maxgens` - number of generations to run simulation.
- `OUTNAME` - prefix to name out files.

Simulation programs other than SLiM may be used to make training data, as long as the output is processed into tensors of the necessary shape.
Given the strict format of the input files, we do not recommend users attempt to generate their own training data from sources other than SLiM.






## Vignette: example workflow

### Custom simulations

Next, we will analyze a theoretical population of *Internecivus raptus* 😱.
Let's assume we have independent estimates from previous studies for the size 
of the species range and the population density: these values are 50x50 km^2, and 6 individuals per square km, respectively.
With values for these nuisance parameters in hand we can design custom training simulations for inferring &#963;.
Lets assume our *a priori* expectation for the dispersal rate in this species is somewhere between 0.2 and 1.5 km/generation;
we want to explore potential dispersal rates in this range.

Below is some bash code to run the simulations (runs for a few minutes, to an hour, depending on threads):

```bash
mkdir temp_wd/TreeSeqs
for i in {1..100}
do
    sigma=$(python -c 'import numpy as np; print(np.random.uniform(0.2,1.5))')
    echo "slim -d SEED=$i -d sigma=$sigma -d K=6 -d mu=0 -d r=1e-8 -d W=50 -d G=1e8 -d maxgens=100 -d OUTNAME=\"'temp_wd/TreeSeqs/output'\" ../disperseNN/SLiM_recipes/bat20.slim" >> temp_wd/sim_commands.txt
    echo temp_wd/TreeSeqs/output_$i.trees >> temp_wd/tree_list.txt
done
parallel -j 20 < temp_wd/sim_commands.txt
```

Note: the carrying capacity in this model, `K`, corresponds roughly to density, but the actual density will fluctuate a bit.



### Training

Before proceeding, we will separate the sims into two groups: (i) training data and (ii) test data.
The latter portion will be held out for testing, later.

```bash
head -50 temp_wd/tree_list.txt > temp_wd/training_trees.txt
tail -50 temp_wd/tree_list.txt > temp_wd/test_trees.txt
```

The training step is computationally intensive and should ideally be run on a computing cluster or cloud system.
The `threads` flag can be altered to use more CPUs for processing tree sequences.
Using 20 dedicated threads (and batch size=20), this step should take several hours; however, if only 1-2 threads are used, the training step will take days.

Our training command will use a similar settings to the above example "Training: tree sequences as input".
Of note, the min and max *n* are both set to 14 because we want to analyze dispersal in a subset of exactly 14 individuals from our empirical data (see below).
We will sample 100 partially overlapping samples of n=14 from each tree sequence for a total training set of size 5000- this is specified via the `num_samples` flag.

```bash
python disperseNN.py \
  --train \
  --min_n 14 \
  --max_n 14 \
  --num_snps 1000 \
  --recapitate True \
  --mutate True \
  --tree_list temp_wd/training_trees.txt \
  --edge_width 1.5 \
  --sampling_width 1 \
  --num_samples 10 \
  --batch_size 20 \
  --threads 1 \
  --max_epochs 1 \
  --out temp_wd/out2 \
  --seed 12345 \
```

Note: here we chose to sample away from the habitat edges by 1.5km.
This is because in the simulation model we artifically reduces survival probability near the edges.




### Testing

Next, we will validate the trained model using the held-out test data.
This command will use a similar set of flags to the above example "Prediction: tree sequences as input" (should take a minute or two to run).

```bash
python disperseNN.py \
  --predict \
  --load_weights temp_wd/out2_model.hdf5 \
  --training_params temp_wd/out2_training_params.npy \
  --recapitate True \
  --mutate True \
  --tree_list temp_wd/test_trees.txt \
  --min_n 14 \
  --edge_width 1.5 \
  --sampling_width 1 \
  --num_pred 10 \
  --batch_size 10 \
  --threads 1 \
  --out temp_wd/out2 \
  --seed 12345
```

The output file `temp_wd/out2_sigma_predictions.txt` shows that our estimates are accurate***, therefore `disperseNN` was successful at learning to estimate &#963;.

```bash
temp_wd/TreeSeqs/output_51.trees 1.13122562 0.51208142
temp_wd/TreeSeqs/output_52.trees 0.3823590837 0.4942408277
temp_wd/TreeSeqs/output_53.trees 0.41219539 0.5039974494
temp_wd/TreeSeqs/output_54.trees 0.4901229428 0.5056558093
temp_wd/TreeSeqs/output_55.trees 0.2506551389 0.4867979277
temp_wd/TreeSeqs/output_56.trees 0.2303818449 0.4834368114
temp_wd/TreeSeqs/output_57.trees 1.0009738369 0.5087425838
temp_wd/TreeSeqs/output_58.trees 1.1115748604 0.5073908072
temp_wd/TreeSeqs/output_59.trees 0.5463572066 0.5131512291
temp_wd/TreeSeqs/output_60.trees 1.2719527705 0.5090078776
```





### VCF prep

If we are satisfied with the performance of the model on the held-out test set, we can run prepare our empirical VCF for inference with `disperseNN`.
This means applying basic filters (e.g. removing indels and non-variants sites) on whatever set of individuals that we want to analyze.
Separately, we want a .locs file with the same prefix as the .vcf.

For demonstration purposes, let's say we want to take a subset of individuals from a particular geographic region, the Scotian Shelf region.
Furthermore, we want to include only a single individual per sampling location; this is important because individuals did not have identical
locations in the training simulations, which might trip up the neural network.
Below are some example commands that might be used to parse the metadata, but these steps will certainly be different for other empirical tables.

```bash
# [these commands are gross; but I want to eventually switch over to simulated data, so these steps will change]
cat Examples/VCFs/iraptus_meta_full.txt | grep "Scotian Shelf - East" | cut -f 4,5 | sort | uniq > temp_wd/templocs
count=$(wc -l temp_wd/templocs | awk '{print $1}')
for i in $(seq 1 $count); do locs=$(head -$i temp_wd/templocs | tail -1); lat=$(echo $locs | awk '{print $1}'); long=$(echo $locs | awk '{print $2}'); grep $lat Examples/VCFs/iraptus_meta_full.txt | awk -v coord=$long '$5 == coord' | shuf | head -1; done > temp_wd/iraptus_meta.txt
cat temp_wd/iraptus_meta.txt  | sed s/"\t"/,/g > temp_wd/iraptus.csv
```

We provide a simple python script for subsetting a VCF for a particular set of individuals, which also filters indels and non-variant sites.

```bash
python Empirical/subset_vcf.py Examples/VCFs/iraptus_full.vcf.gz temp_wd/iraptus.csv temp_wd/iraptus.vcf 0 1
```

Last, build a .locs file:

```bash
count=$(zcat temp_wd/iraptus.vcf.gz | grep -v "##" | grep "#" | wc -w)
for i in $(seq 10 $count); do id=$(zcat temp_wd/iraptus.vcf.gz | grep -v "##" | grep "#" | cut -f $i); grep -w $id temp_wd/iraptus.csv; done | cut -d "," -f 4,5 | sed s/","/"\t"/g > temp_wd/iraptus.locs
gunzip temp_wd/iraptus.vcf.gz
```

### Empirical inference

Finally, we can predict predict &#963; from the subsetted VCF (should take less than 30s to run):

```bash
python disperseNN.py \
  --predict \
  --load_weights temp_wd/out2_model.hdf5 \
  --training_params temp_wd/out2_training_params.npy \
  --empirical temp_wd/iraptus \
  --num_reps 10 \
  --out temp_wd/out3 \
  --seed 12345
```

Note: `num_reps`, here, specifies how many bootstrap replicates to perform, that is, how many seperate draws of 1000 SNPs to use as inputs for prediction.

The final empirical results are stored in: temp_wd/out3_predictions.txt

```bash
temp_wd/iraptus_0 0.4790744392
temp_wd/iraptus_1 0.4782159438
temp_wd/iraptus_2 0.4752711311
temp_wd/iraptus_3 0.4757308299
temp_wd/iraptus_4 0.4763104592
temp_wd/iraptus_5 0.4740976943
temp_wd/iraptus_6 0.4711097443
temp_wd/iraptus_7 0.4765035801
temp_wd/iraptus_8 0.4711986949
temp_wd/iraptus_9 0.4780693254
```

## References

Battey CJ, Ralph PL, Kern AD. Space is the place: effects of continuous spatial structure on analysis of population genetic data. Genetics. 2020 May 1;215(1):193-214.
