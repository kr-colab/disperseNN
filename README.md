# disperseNN

## Install Requirements
`pip install -r requirements.txt` should cover you

## Example Commands (only the first one is expected to work right now)

### Predict with empirical data, e.g. RADseq
python /home/chriscs/kernlab/Maps/Maps/disperseNN.py --predict --out out1 --num_snps 5000 --training_mean -0.9874806682910889 --training_sd 1.8579295139087375 --max_n 100 --mu 1e-8 --seed 12345 --phase 1 --polarize 2 --load_weights Saved_models/out136_2400.12_model.hdf5 --empirical ExampleVCFs/halibut --bootstrap_reps_radseq 1000 --num_pred 10

### Train with tree sequences
python /home/chriscs/kernlab/Maps/Maps/disperseNN.py --out out1 --num_snps 5000 --target_list Boxes34/map_list.txt --max_epochs 100 --validation_split 0.2 --num_pred 0 --batch_size 40 --threads 1 --min_n 100 --max_n 100 --genome_length 100000000 --mu 1e-8 --seed 12345 --tree_list Boxes34/tree_list.txt --recapitate False --mutate True --phase 2 --map_width 50 --sampling_width 1 --on_the_fly 50 --edge_width 3 --train

### Train with pre-processed tensors
python /home/chriscs/kernlab/Maps/Maps/disperseNN.py --out out1 --num_snps 5000 --target_list Hierarchical_2400_unphased/map_list.txt --max_epochs 100 --validation_split 0.2 --batch_size 40 --threads 1 --min_n 100 --max_n 100 --genome_length 100000000 --mu 1e-8 --seed 12345 --samplewidth_list Hierarchical_2400_unphased/samplewidth_list.txt --geno_list Hierarchical_2400_unphased/geno_list.txt --loc_list Hierarchical_2400_unphased/loc_list.txt --pos_list Hierarchical_2400_unphased/pos_list.txt --recapitate False --mutate True --phase 1 --preprocess --train --map_width 50

### Predict with tree sequences
python /home/chriscs/kernlab/Maps/Maps/disperseNN.py --out out1 --num_snps 5000 --training_targets Boxes54/map_list_raw.txt --max_epochs 100 --validation_split 0.2 --batch_size 10 --threads 10 --min_n 50 --max_n 50 --genome_length 100000000 --mu 1e-8 --seed 12345 --recapitate False --mutate True --phase 2 --sampling_width 1 --on_the_fly 50 --target_list tempmaps --width_list tempwidths --tree_list temptrees --load_weights Boxes54/out142_boxes54.2_model.hdf5 --predict --num_pred 10

### Predict with pre-processed tensors
python /home/chriscs/kernlab/Maps/Maps/disperseNN.py --out out1 --num_snps 5000 --training_targets temptargets --max_epochs 100 --validation_split 0.2 --batch_size 1 --threads 1 --min_n 10 --max_n 100 --genome_length 100000000 --mu 1e-8 --seed 12345 --samplewidth_list tempsamplewidths --geno_list tempgenos --loc_list templocs --pos_list temppos --target_list temptargets --recapitate False --mutate True --phase 1 --preprocess --load_weights /home/chriscs/kernlab/Maps/Boxes66/out136_unphased.17_resumed1_model.hdf5 --predict --num_pred 10

### Predict with empirical data: using pre-processed tensors for genomic windows
python /home/chriscs/kernlab/Maps/Maps/disperseNN.py --num_snps 5000 --training_targets Hierarchical_hier1/map_list.txt --validation_split 0.2 --num_pred 10 --batch_size 1 --recapitate False --mutate True --load_weights ../Maps/Important_saved_models/out136_unphased.17_model.hdf5 --geno_list ../AG1000_phase3/geno_list_unphased.txt --pos_list ../AG1000_phase3/pos_list_uphased.txt --empirical ../AG1000_phase3/cameroon_chrY_unplaced_set1 --preprocess --predict --out out1 --phase 1 --max_n 100

### Minimal filtering for RAD data
1. figure out what samples you want to include. E.g. maybe all samples in the vcf, maybe a geographic cluster.
   For example: oyster RADseq
   oyster_meta.txt # table 1 from paper
   count=$(wc -l oyster_meta.txt | awk '{print $1}')
   for i in $(seq 1 $count); do pop=$(head -$i oyster_meta.txt | tail -1 | cut -f 2); cat population_map.txt | awk -v pop2=$pop '$2 == pop2' | shuf | head -1 | cut -f 1; done > oyster_samples.txt
   for thing in $(cat oyster_samples.txt); do p=$(echo $thing | cut -d "_" -f 1); coords=$(cat oyster_meta.txt | awk -v pop=$p '$2 == pop' | cut -f 3,4); echo $thing $coords; done | sed s/" "/","/ > oyster.csv

2. subset_vcf.py— this takes only the samples you want, and filters at the same time (replaces filter_vcf.py)
   python ../../Maps/Empirical/mat2vcf.py population_map.txt matrix012_8246SNPs_NEUTRAL_IMPUTED.txt > matrix012_8246SNPs_NEUTRAL_IMPUTED.vcf 
   gzip matrix012_8246SNPs_NEUTRAL_IMPUTED.vcf 
   python ../../Maps/Empirical/subset_vcf.py matrix012_8246SNPs_NEUTRAL_IMPUTED.vcf.gz oyster.csv oyster.vcf 0 1

3. make a locs file—same order as samples in VCF.
   count=$(zcat oyster.vcf.gz | grep -v "##" | grep "#" | wc -w)
   for i in $(seq 10 $count); do id=$(zcat oyster.vcf.gz | grep -v "##" | grep "#" | cut -f $i); grep -w $id oyster.csv; done | cut -d "," -f 2,3 | sed s/","/"\t"/g > oyster.locs

4. should be good to go!

### Pre-process genomic windows from WGS data
bash pipe_chroms.sh cameroon.vcf.gz cameroon chrom_list.txt cameroon.csv
bash vcf2windows.sh cameroon 200000 Windows_200Kb_allChroms 5000 chrom_list.txt
