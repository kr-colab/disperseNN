# disperseNN

## Install Requirements
`conda create --name dispersenn python=3.8`

`pip install -r requirements.txt` should cover you

## Brief instructions
`disperseNN` has two modes: 
* prediction:   specify the `--predict` flag
    - Input types:
        1. VCF
        2. tree sequences
        3. pre-processed tensors
* training:     by specifying the `--train` flag
    - Input types:
      	1. tree sequences
        2. pre-processed tensors

Within each mode, different types of input data may be specified, 

## Example Commands
The different input types will require different sets of command line parameters. Below are the different input types with example commands.

### Predict with empirical data, e.g. RADseq
python disperseNN.py --predict --out out1 --num_snps 5000 --training_mean -0.9874806682910889 --training_sd 1.8579295139087375 --max_n 100 --mu 1e-8 --seed 12345 --phase 1 --polarize 2 --load_weights Saved_models/out136_2400.12_model.hdf5 --empirical ExampleVCFs/halibut --bootstrap_reps_radseq 1000 --num_pred 10
