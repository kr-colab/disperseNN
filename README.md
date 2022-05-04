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
- empirical: this flag is specific to analyzing VCFs. Give it the prefix for the .vcf and .locs files (i.e. no '.vcf' or '.locs')
- max_n: sample size
- num_snps: number of snps to analyze. This number equals num_snps in the loaded model, but is probably fewer than the VCF lines.
- phase: '1' for unphased, '2' for phased 
- polarize: '1' for polarized, '2' for unpolarized
- load_weights: saved model or weights to load
- training_mean: mean from the training targets
- training_sd: standard deviation from the training targets
- num_pred: number of datasets to predict; here, the number of bootstrap replicates
- out: output prefix
- seed: random number seed


### Prediction: tree sequences as input
```
python disperseNN.py --predict --min_n 50 --max_n 50 --num_snps 5000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --on_the_fly 50 --tree_list tree_list.txt --target_list target_list.txt --width_list width_list.txt --load_weights Saved_models/out136_2400.12_model.hdf5 --training_mean -0.9874806682910889 --training_sd 1.8579295139087375 --num_pred 1000 --batch_size 10 --threads 10 --out out1 --seed 123451
```

### Prediction: using pre-processed tensors
```
python disperseNN.py --predict --min_n 50 --max_n 50 --num_snps 5000 --genome_length 100000000 --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --on_the_fly 50 --preprocess --geno_list geno_list.txt --loc_list loc_list.txt --pos_list pos_list.txt --samplewidth_list sample_widths.txt --target_list target_list.txt --load_weights Saved_models/out136_2400.12_model.hdf5 --training_mean -0.9874806682910889 --training_sd 1.8579295139087375 --num_pred 1000 --batch_size 10 --threads 10 --out out1 --seed 123451
```

python /home/chriscs/kernlab/Maps/Maps/disperseNN.py --out out1 --num_snps 5000 --training_targets temptargets --max_epochs 100 --validation_split 0.2 --batch_size 1 --threads 1 --min_n 10 --max_n 100 --genome_length 100000000 --mu 1e-8 --seed 12345 --samplewidth_list tempsamplewidths --geno_list tempgenos --loc_list templocs --pos_list temppos --target_list temptargets --recapitate False --mutate True --phase 1 --preprocess --load_weights /home/chriscs/kernlab/Maps/Boxes66/out136_unphased.17_resumed1_model.hdf5 --predict --num_pred 10





### Training: tree sequences as input
```
```

### Training: with pre-processed tensors
```
```

## Simulation
The authors of `disperseNN` used the slim recipe _____ to generate training data (tree sequences).

Simulation programs other than SLiM may be used to make training data, as long as the output is processed into tensors of the necessary shape. 