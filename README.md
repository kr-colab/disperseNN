Be sure to check out our newer method, [`disperseNN2`](https://github.com/kr-15
colab/disperseNN2).


# disperseNN

- [disperseNN](#dispersenn)
  - [Install requirements](#install-requirements)
  - [Overview](#overview)
  - [The pre-trained model](#the-pre-trained-model)
  - [Instructions with example commands](#instructions-with-example-commands)
    - [Simulation](#simulation)
    - [Training](#training)
    - [Prediction: tree sequences as input](#prediction-tree-sequences-as-input)
  - [Vignette: example workflow](#vignette-example-workflow)
    - [Custom simulations](#custom-simulations)
    - [Training](#training)
    - [Testing](#testing)
    - [VCF prep](#vcf-prep)
    - [Empirical inference](#empirical-inference)
  - [References](#references)

`disperseNN` is a machine learning framework for predicting &#963;, the expected per generation displacement distance between offspring and their parent(s), from genetic variation data.
`disperseNN` uses training data generated from simulations.
See [Smith et al.](https://www.biorxiv.org/content/10.1101/2022.08.25.505329v3) for a comprehensive description of the method.





## Install requirements

First clone this repository:

```bash
git clone https://github.com/chriscrsmith/disperseNN.git
cd disperseNN/
```

Next create a `conda` environment using our provided `yaml` file. 
This will create a virtual environment, and then install all the
dependencies. 

```bash
mamba env create -f conda_env.yml
```

After `conda` has done its thing, activate the new environment
to use `disperseNN`
```bash
conda activate disperseNN
```





## Overview

`disperseNN` has two modes:


### 1. Prediction

  Command line flag:  `--predict`

  Input options:
  * VCF
  * tree sequences
  * pre-processed numpy arrays

  
### 2. Training

  Command line flag:  `--train`

  Input options:  
  * tree sequences
  * pre-processed numpy arrays

Within each mode- `predict` or `train`- you may specify different types of input data, each requiring its own set of additional command line parameters. More details are below.





## The pre-trained model
While `disperseNN` can be trained from scratch, we recommend trying the pre-trained model provided in this repository first
(see [Smith et al.](https://kr-colab.github.io/) for model details).
Before handing an empirical VCF to `disperseNN`, it should undergo basic filtering steps to remove non-variant sites and indels; rare variants should be left in.
Furthermore, the VCF should include only the individuals that you intend to analyze.
At least 5,000 SNPs are required for the current model,
but more than 5,000 SNPs can be left in the VCF because `disperseNN` will draw a random subset.
Last, a .locs file should be prepared with two columns corresponding to the latitude and longitude of each individual.

To run the example commands, begin by setting up a new working directory:

```bash
mkdir -p temp_wd/
```

Below is an example command for estimating &#963; from a VCF file using a pre-trained model (should take <30s to run).

```bash
python disperseNN.py \
  --predict \
  --load_weights Saved_models/pretrained082522_model.hdf5 \
  --training_params Saved_models/pretrained082522_training_params.npy \
  --empirical Examples/VCFs/halibut \
  --num_reps 10 \
  --out temp_wd/out_vcf \
  --seed 12345
```

Explanation of command line values:
- `load_weights`: this specifies the path to the saved model.
The above command points to a particular pre-trained model, `pretrained082522_model.hdf5`,
but instructions for how to train a new model from scratch are described below.
- `training_params`: a single file with several parameters used to train the above model. Included are: the mean and standard deviation used to normalize the training targets, and the number of SNPs and maximum sample size used to train the above model. In this case, the number of SNPs is 5,000 and the max sample size is 100.
- `empirical`: this flag is specific to analyzing empirical VCFs. Give it the shared prefix for the .vcf and .locs files (i.e. without '.vcf' or '.locs')
- `num_reps`: number of repeated draws (of 5,000 SNPs in this case) from the VCF, with a single prediction per subset.
- `out`: output prefix.
- `seed`: random number seed.

In addition to printing information about the model architecture to standard output, this command will also create a new file, `temp_wd/out_vcf_predictions.txt`, containing:

```bash
Examples/VCFs/halibut_0 4.008712555
Examples/VCFs/halibut_1 2.7873949732
Examples/VCFs/halibut_2 3.7759448146
Examples/VCFs/halibut_3 3.2785118587
Examples/VCFs/halibut_4 2.6940501913
Examples/VCFs/halibut_5 2.8515263298
Examples/VCFs/halibut_6 3.1886536211
Examples/VCFs/halibut_7 2.5544670147
Examples/VCFs/halibut_8 2.7795463315
Examples/VCFs/halibut_9 3.9511181921
```

Where each line is one of the 10 predictions of &#963; using a random subset of 5K SNPs.





## Instructions with example commands


### Simulation

The pre-trained model provided may not be appropriate for your data.
In this case, it is possible to train a new model from scratch from a simulated training set.
We use the SLiM recipe `SLiM_recipes/bat20.slim` to generate training data (tree sequences).
The model is adapted from [Battey et al. (2020)](https://doi.org/10.1534/genetics.120.303143),
but certain model parameters are specified on the command line.

As a demonstration, see the below example command (this simulation may run for several minutes, but feel free to kill it with ctrl-C; we don't need this output for any downstream steps):

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

Command line arguments are passed to SLiM using the `-d` flag followed by the variable name as it appears in the recipe file.

- `SEED` - a random seed to reproduce the simulation results.
- `sigma` - the dispersal parameter.
- `K` - carrying capacity.
- `mu` - per base per genertation mutation rate.
- `r` -  per base per genertation recombination rate.
- `W` - the height and width of the geographic spatial boundaries.
- `G` - total size of the simulated genome.
- `maxgens` - number of generations to run simulation.
- `OUTNAME` - prefix to name output files.

Note: after running SLiM for a fixed number of generations, the simulation is still not complete, as many trees will likely not have coalesced still. We usually finish the simulation backwards in time using mspime, e.g., "recapitation". This can either be done before training (recommended; see example below) or during training using the `recapitate` flag. 

Simulation programs other than SLiM may be used to make training data, as long as the output is processed into tensors of the necessary shape. 
Given the strict format of the input files, we do not recommend users attempt to generate training data from sources other than SLiM.

In practice, we use 1000 or more simulations like the above one for a given training run (and then subsample from each simulation to achieve a training set of 50,000). Ideally simulations should be run on a high performance computing cluster.

### Training

Below is an example command for the training step.
This example uses tree sequences as input (again, feel free to kill this command).

```bash
python disperseNN.py \
  --train \
  --tree_list Examples/tree_list1.txt \
  --mutate True \
  --min_n 10 \
  --max_n 10 \
  --edge_width 3 \
  --sampling_width 1 \
  --num_snps 1000 \
  --repeated_samples 100 \
  --batch_size 10 \
  --threads 1 \
  --max_epochs 10 \
  --seed 12345 \
  --out temp_wd/out1
```

- `tree_list`: list of paths to the tree sequences. &#963; values and habitat widths are extracted directly from the tree sequence.
- `mutate`: add mutations to the tree sequence until the specified number of SNPs are obtained (5,000 in this case, specified inside the training params file).
- `min_n`: specifies the minimum sample size. 
- `max_n`: paired with `min_n` to describe the range of sample sizes to drawn from. Set `min_n` equal to `max_n` to use a fixed sample size.
- `edge_width`: this is the width of edge to 'crop' from the sides of the habitat. In other words, individuals are sampled `edge_width` distance from the sides of the habitat.
- `sampling_width`: samples individuals from a restricted sampling window with width between 0 and 1, in proportion to the habitat width, after excluding edges.
- `num_snps`: the number of SNPs to use as input for the CNN.
- `repeated_samples`: this is the number of repeated draws of `n` individuals to take from each tree sequence. This let's us get away with fewer simulations.
- `batch_size`: for the data generator. We find that batch_size=40 works well if the training set is larger.
- `threads`: number of threads to use for multiprocessing during the data generation step.
- `max_epochs`: maximum number of epochs to train for.
- `seed`: random number seed.
- `out`: output prefix.

This run will eventually print the training progress to stdout, while the model weights are saved to `temp_wd/out1_model.hdf5`.

Also, this example command is small-scale; in practice, you will need a training set of maybe 50,000, and you will want to train for longer than 10 epochs. 

### Prediction: tree sequences as input

If you want to predict &#963; in simulated tree sequences, such as those generated by `msprime` and `SLiM`, the predict command will look a bit different than predicting with a VCF as input. Below is an example command (should take <30s to run). Each command line flag is described in the preceding examples.


```bash
python disperseNN.py \
  --predict \
  --load_weights Saved_models/pretrained082522_model.hdf5 \
  --training_params Saved_models/pretrained082522_training_params.npy \
  --tree_list Examples/tree_list1.txt \
  --mutate True \
  --min_n 10 \
  --edge_width 3 \
  --sampling_width 1  \
  --seed 12345 \
  --out temp_wd/out_treeseq
```

Similar to the earlier prediction example, this will generate a file called `temp_wd/out_treeseq_predictions.txt` containing:

```bash
Examples/TreeSeqs/output_2_recap.trees 0.5914545564 0.6582331812
Examples/TreeSeqs/output_3_recap.trees 0.3218814158 0.3755014635
Examples/TreeSeqs/output_1_recap.trees 0.3374337601 0.4073884732
Examples/TreeSeqs/output_5_recap.trees 0.2921853737 0.2047981935
Examples/TreeSeqs/output_4_recap.trees 0.277020769 0.3208989912
```

Here, the second and third columns contain the true and predicted &#963; for each simulation.





## Vignette: example workflow

This vignette will walk through an example workflow with more verbose instructions, particularly for the intermediate data-organizing steps.

### Custom simulations

We will analyze a theoretical population of *Internecivus raptus* 😱.
Let's assume we have independent estimates from previous studies for the size 
of the species range and the population density: these values are 50x50 km^2, and 6 individuals per square km, respectively.
With values for these nuisance parameters in hand we can design custom training simulations for inferring &#963;.
Lets assume our *a priori* expectation for the dispersal rate in this species is somewhere between 0.2 and 1.5 km/generation;
we want to explore potential dispersal rates in this range.

Below is some bash code to run the simulations. Threads can be adjusted via the `-j` flag to `parallel` (runs for a few minutes, to an hour, depending on threads):

```bash
mkdir temp_wd/TreeSeqs
for i in {1..100}
do
    sigma=$(python -c 'from scipy.stats import loguniform; print(loguniform.rvs(0.2,1.5))')
    echo "slim -d SEED=$i -d sigma=$sigma -d K=6 -d mu=0 -d r=1e-8 -d W=50 -d G=1e8 -d maxgens=100 -d OUTNAME=\"'temp_wd/TreeSeqs/output'\" ../disperseNN/SLiM_recipes/bat20.slim" >> temp_wd/sim_commands.txt
done
parallel -j 2 < temp_wd/sim_commands.txt
```

Note: the carrying capacity in this model, `K`, corresponds roughly to density, but the actual density will fluctuate a bit.

Next we will recapitate the tree sequences. We choose to do this step up front, as it makes the training step much faster. However you may choose to skip this preliminary step, and instead recapitate on-the-fly during training using `--recapitate True`, which has the benefit of adding variation to the training set. 

This code block recapitates (runs for a few minutes, to an hour, depending on threads):

```bash
for i in {1..100};
do
    echo "python -c 'import tskit; from process_input import *; ts=tskit.load(\"temp_wd/TreeSeqs/output_$i.trees\"); ts=recapitate(ts,1e-8,$i); ts.dump(\"temp_wd/TreeSeqs/output_$i"_"recap.trees\")'" >> temp_wd/recap_commands.txt
    echo temp_wd/TreeSeqs/output_$i"_"recap.trees >> temp_wd/tree_list.txt
done   
parallel -j 2 < temp_wd/recap_commands.txt
```




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
We will sample 100 partially overlapping samples of n=14. 
This is specified via the `repeated_samples` flag, 
and will result in a total training set of size 5,000.

```bash
python disperseNN.py \
  --train \
  --min_n 14 \
  --max_n 14 \
  --num_snps 1000 \
  --mutate True \
  --tree_list temp_wd/training_trees.txt \
  --edge_width 1.5 \
  --sampling_width 1 \
  --repeated_samples 10 \
  --batch_size 20 \
  --threads 1 \
  --max_epochs 1 \
  --out temp_wd/out2 \
  --seed 12345
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
  --mutate True \
  --tree_list temp_wd/test_trees.txt \
  --min_n 14 \
  --edge_width 1.5 \
  --sampling_width 1 \
  --batch_size 10 \
  --threads 1 \
  --out temp_wd/out2 \
  --seed 12345
```

The output file `temp_wd/out2_predictions.txt` shows our estimates, which are not expected to be very good after such a small training run.

```bash
temp_wd/TreeSeqs/output_84_recap.trees 0.3967863343 0.4447612993
temp_wd/TreeSeqs/output_98_recap.trees 0.4485952108 0.4456993015
temp_wd/TreeSeqs/output_77_recap.trees 0.2158582993 0.4422532732
temp_wd/TreeSeqs/output_68_recap.trees 0.3567543799 0.4427506462
temp_wd/TreeSeqs/output_76_recap.trees 0.2504898583 0.4447698645
temp_wd/TreeSeqs/output_59_recap.trees 1.2755786915 0.4527958429
temp_wd/TreeSeqs/output_99_recap.trees 0.9861785139 0.4497781552
temp_wd/TreeSeqs/output_69_recap.trees 1.0079008897 0.4538938886
temp_wd/TreeSeqs/output_82_recap.trees 0.3062711445 0.4487468518
temp_wd/TreeSeqs/output_80_recap.trees 0.7506339226 0.4483650639
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
