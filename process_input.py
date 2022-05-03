
import os
import numpy as np
from read_input import *
import sys

# using my map projection in R to rescale the locs...                                
def project_locs(locs,out,seed):
    tmp_name = "." + out + "." + str(seed)
    with open(tmp_name, "w") as outfile:
        for i in range(len(locs)):
            outfile.write("\t".join(locs[i]) + "\n")
    print("projecting locations...")
    comm = "Rscript Empirical/lat2km_v3.R " + tmp_name # need to make this more robust
    os.system(comm)
    new_locs = list(np.array(read_locs(tmp_name+"_proj")).astype(float)) # (shenanigans to get float)
    new_locs = np.array(new_locs)
    comm = "rm "+tmp_name
    os.system(comm)      
    comm = "rm "+tmp_name+"_proj"
    os.system(comm)              
    return new_locs

# rescale locs to (0,1)                                                      
def rescale_locs(locs):
    locs0=np.array(locs)
    minx = min(locs0[:,0])
    maxx = max(locs0[:,0])
    miny = min(locs0[:,1])
    maxy = max(locs0[:,1])
    x_range = maxx - minx
    y_range = maxy - miny
    sample_width = max(x_range,y_range) # re-define width to be this distance
    locs0[:,0] =(locs0[:,0] - minx) / (maxx - minx) # rescale to (0,1)       
    locs0[:,1] =(locs0[:,1] - miny) / (maxy - miny)
    if x_range > y_range:
        locs0[:,1] *= (y_range / x_range) # scale down y, proportional to x. 
    elif y_range > x_range:
        locs0[:,0] *= (x_range / y_range) # scale down y, proportional to x. 
    test_locs = locs0.T
    sample_width = np.array(sample_width)
    return test_locs,sample_width

def pad_locs(locs,max_n):
    padded = np.zeros((2,max_n))
    n = locs.shape[1]
    padded[:,0:n] = locs
    return padded

# pre-processing rules:                                                                                                                           
#     1 biallelic change the alelles to 0 and 1 before inputting.
#     2. no missing data: filter or impute.                                                                                                       
#     3. ideally no sex chromosomes, and only look at one sex at a time.                                                                          
def vcf2genos(vcf_path,max_n,num_snps,phase):
    geno_mat,pos_list=[],[]
    vcf = open(vcf_path, "r")
    current_chrom = "XX"
    baseline_pos = 0
    previous_pos = 0
    for line in vcf:
        if line[0] != "#":
            newline = line.strip().split("\t")
            genos = []
            for field in range(9,len(newline)):
                geno = newline[field].split(":")[0].split("/")
                geno = [int(geno[0]),int(geno[1])]
                if phase == 1:
                    genos.append(sum(geno)) # collapsed genotypes   
                elif phase == 2:
                    geno = [min(geno), max(geno)] 
                    genos.append(geno[0])
                    genos.append(geno[1])
                else:
                    print("problem")
                    exit()
            for i in range((max_n*phase)-len(genos)): # pad with 0s                                                                                  
                genos.append(0)
            geno_mat.append(genos)

            # deal with genomic position                                                                                                          
            chrom = newline[0]
            pos = int(newline[1])
            if chrom == current_chrom:
                previous_pos = int(pos)
            else:
                current_chrom = str(chrom)
                baseline_pos = int(previous_pos) + 10000 # skipping 10kb between chroms/scaffolds (also skipping 10kb before first snp, currently)
            current_pos = baseline_pos + pos
            pos_list.append(current_pos)

    # check if enough snps
    if len(geno_mat) < num_snps:
        print("not enough snps")
        exit()
    if len(geno_mat[0]) < (max_n*phase):
        print("not enough samples")
        exit()

    # sample snps                                                                                                                                 
    geno_mat = np.array(geno_mat)
    pos_list = np.array(pos_list)
    mask = [True]*num_snps + [False]*(geno_mat.shape[0] - num_snps)
    np.random.shuffle(mask)
    geno_mat = geno_mat[mask,:]
    pos_list = pos_list[mask]

    # rescale positions                                                                                                                           
    pos_list = pos_list / (max(pos_list)+1) # +1 to avoid prop=1.0 

    return geno_mat,pos_list



### main
def main():
    vcf_path = sys.argv[1]
    max_n = int(sys.argv[2])
    num_snps = int(sys.argv[3])
    outname = sys.argv[4]
    phase = int(sys.argv[5])   
    geno_mat,pos_list = vcf2genos(vcf_path,max_n,num_snps,phase)
    np.save(outname+".genos", geno_mat)
    np.save(outname+".pos", pos_list)
    

if __name__ == "__main__": main()






