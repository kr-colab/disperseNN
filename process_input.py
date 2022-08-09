# helper utils for processing empirical input data

import numpy as np
import sys
from geopy import distance
import random

# get sampling width


def project_locs(coords):
    n = len(coords)
    sampling_width = 0
    for i in range(0, n-1):
        for j in range(i+1, n):
            # ellipsoid='WGS-84' by default
            d = distance.distance(coords[i, :], coords[j, :]).km
            if d > sampling_width:
                sampling_width = float(d)
    return sampling_width


# pad locations with zeros
def pad_locs(locs, max_n):
    padded = np.zeros((2, max_n))
    n = locs.shape[1]
    padded[:, 0:n] = locs
    return padded


# pre-processing rules:
#     1 biallelic change the alelles to 0 and 1 before inputting.
#     2. no missing data: filter or impute.
#     3. ideally no sex chromosomes, and only look at one sex at a time.
def vcf2genos(vcf_path, max_n, num_snps, phase):
    geno_mat = []
    vcf = open(vcf_path, "r")
    current_chrom = None
    for line in vcf:
        if line[0:2] == "##":
            pass
        elif line[0] == "#":
            header = line.strip().split("\t")
            if max_n == None:  # option for getting sample size from vcf
                max_n = len(header)-9
        else:
            newline = line.strip().split("\t")
            genos = []
            for field in range(9, len(newline)):
                geno = newline[field].split(":")[0].split("/")
                geno = [int(geno[0]), int(geno[1])]
                if phase == 1:
                    genos.append(sum(geno))
                elif phase == 2:
                    genos.append(geno[0])
                    genos.append(geno[1])
                else:
                    print("problem")
                    exit()
            for i in range((max_n * phase) - len(genos)):  # pad with 0s
                genos.append(0)
            geno_mat.append(genos)

    # check if enough snps
    if len(geno_mat) < num_snps:
        print("not enough snps")
        exit()
    if len(geno_mat[0]) < (max_n * phase):
        print("not enough samples")
        exit()

    # sample snps
    geno_mat = np.array(geno_mat)
    return geno_mat[np.random.choice(geno_mat.shape[0], num_snps, replace=False), :]


# calculate isolation by distance
def ibd(genos, coords, phase, num_snps):

    # subset for n samples (avoiding padding-zeros)
    n = 0
    for i in range(genos.shape[1]):
        reverse_index = genos.shape[1]-i-1
        if len(set(genos[:, reverse_index])) > 1:
            n += reverse_index
            break
    n += 1  # for 0 indexing
    if phase == 2:
        n = int(n/2)
    genos = genos[:, 0:n*phase]  # *** maybe don't need to subset, as long as we know n?  

    # if collapsed genos, make fake haplotypes for calculating Rousset's statistic
    if phase == 1:
        geno_mat2 = []
        for i in range(genos.shape[1]):
            geno1, geno2 = [], []
            for s in range(genos.shape[0]):
                combined_geno = genos[s, i]
                if combined_geno == 0.0:
                    geno1.append(0)
                    geno2.append(0)
                elif combined_geno == 2:
                    geno1.append(1)
                    geno2.append(1)
                elif combined_geno == 1:
                    alleles = [0, 1]
                    # assign random allele to each haplotype
                    geno1.append(alleles.pop(random.choice([0, 1])))
                    geno2.append(alleles[0])
                else:
                    print("bug", combined_geno)
                    exit()
            geno_mat2.append(geno1)
            geno_mat2.append(geno2)
        geno_mat2 = np.array(geno_mat2)
        genos = geno_mat2.T

    # denominator for "a"
    locus_specific_denominators = np.zeros((num_snps))
    P = (n*(n-1))/2  # number of pairwise comparisons
    for i1 in range(0, n-1):
        X11 = genos[:, i1*2]
        X12 = genos[:, i1*2+1]
        X1_ave = (X11+X12)/2  # average allelic does within individual-i
        for i2 in range(i1+1, n):
            X21 = genos[:, i2*2]
            X22 = genos[:, i2*2+1]
            X2_ave = (X21+X22)/2
            #
            SSw = (X11-X1_ave)**2 + (X12-X1_ave)**2 + \
                (X21-X2_ave)**2 + (X22-X2_ave)**2
            locus_specific_denominators += SSw
    locus_specific_denominators = locus_specific_denominators / (2*P)
    denominator = np.sum(locus_specific_denominators)

    # numerator for "a"
    gendists = []
    for i1 in range(0, n-1):
        X11 = genos[:, i1*2]
        X12 = genos[:, i1*2+1]
        X1_ave = (X11+X12)/2  # average allelic does within individual-i
        for i2 in range(i1+1, n):
            X21 = genos[:, i2*2]
            X22 = genos[:, i2*2+1]
            X2_ave = (X21+X22)/2
            #
            SSw = (X11-X1_ave)**2 + (X12-X1_ave)**2 + \
                (X21-X2_ave)**2 + (X22-X2_ave)**2
            Xdotdot = (X11+X12+X21+X22)/4  # average allelic dose for the pair
            # a measure of between indiv
            SSb = (X1_ave-Xdotdot)**2 + (X2_ave-Xdotdot)**2
            locus_specific_numerators = ((2*SSb)-SSw) / 4
            numerator = np.sum(locus_specific_numerators)
            a = numerator/denominator
            gendists.append(a)

    # geographic distance
    geodists = []
    for i in range(0, n-1):
        for j in range(i+1, n):
            d = distance.distance(coords[i, :], coords[j, :]).km
            d = np.log(d)
            geodists.append(d)

    # regression
    from scipy import stats
    geodists = np.array(geodists)
    gendists = np.array(gendists)
    b = stats.linregress(geodists, gendists)[0]
    r = stats.pearsonr(geodists, gendists)[0]
    r2 = r**2
    Nw = (1 / b)
    print("IBD r, r^2, slope, Nw:", r, r2, b, Nw)


# main
def main():
    vcf_path = sys.argv[1]
    max_n = sys.argv[2]
    if max_n == "None":
        max_n = None
    else:
        max_n = int(max_n)
    num_snps = int(sys.argv[3])
    outname = sys.argv[4]
    phase = int(sys.argv[5])
    geno_mat = vcf2genos(vcf_path, max_n, num_snps, phase)
    np.save(outname + ".genos", geno_mat)


if __name__ == "__main__":
    main()
