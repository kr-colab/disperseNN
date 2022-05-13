# helper utils for processing empirical input data

import numpy as np
import sys
from geopy import distance


# project locations from ellipsoid (lat,long) to square map (km)
def project_locs(coords):

    # find min/max lat and long
    coords = np.array(coords)
    min_lat = min(coords[:,0])
    max_lat = max(coords[:,0])
    min_long = min(coords[:,1]) 
    max_long = max(coords[:,1])
    lat_range = max_lat - min_lat # latitudinal range
    long_range = max_long - min_long

    # quick check to make sure the samples don't span 180 degrees, or the 180th meridian
    if abs(lat_range) > 180 or abs(long_range) > 180:
        print("samples coords span over 180 degrees or 180th meridian; the code isn't ready to deal with that")
        exit()

    # find a good width for the sampling window
    distA = distance.distance([min_lat,min_long], [max_lat,min_long]).km # confirmed ellipsoid='WGS-84' by default
    distB = distance.distance([min_lat,max_long], [max_lat,max_long]).km
    y_range = max(distA,distB) # range in km
    distC = distance.distance([min_lat,min_long], [min_lat,max_long]).km
    distD = distance.distance([max_lat,min_long], [max_lat,max_long]).km
    x_range = max(distC,distD)
    sampling_width = max([y_range,x_range])

    # rescale locs to (0,1)
    coords[:, 0] = (coords[:, 0] - min_lat) / lat_range # use latitudinal range, here.
    coords[:, 1] = (coords[:, 1] - min_long) / long_range

    # restore aspect ratio
    if   x_range > y_range: # use km range, here.
        coords[:, 0] *= y_range / x_range 
    elif x_range < y_range:
        coords[:, 1] *= x_range / y_range
    coords = coords.T

    return coords, sampling_width
        
    ##### Alternatively: try to correct longitudinal stretch continuously #####
    # # set bottom left corner of sampling window
    # corner_bl = [min_lat, min_long]

    # # bottom right corner: draw line S distance, same lat
    # corner_br = list(corner_bl) # starting on top of the bottom left point
    # dist_bottom = 0
    # precision = 0.00001
    # while dist_bottom < S:
    #     corner_br[1] += precision
    #     dist_bottom = distance.distance(corner_bl, corner_br).km

    # # top corners: draw both sides simultaneously
    # b=0
    # corner_tl = list(distance.distance(kilometers=S).destination(corner_bl, bearing=0))[0:2] # third val is altitude 
    # corner_tr = list(distance.distance(kilometers=S).destination(corner_br, bearing=0))[0:2]
    # dist_top = distance.distance(corner_tl, corner_tr).km
    # if (dist_bottom - dist_top) > 0: # e.g. northern hemisphere
    #     while dist_top < S:
    #         b += precision
    #         corner_tl = list(distance.distance(kilometers=S).destination(corner_bl, bearing=-b))[0:2]
    #         corner_tr = list(distance.distance(kilometers=S).destination(corner_br, bearing=b))[0:2]
    #         dist_top = distance.distance(corner_tl, corner_tr).km
    # else: # e.g. southern hemisphere
    #     while dist_top > S:
    #         b += precision
    #         corner_tl = list(distance.distance(kilometers=S).destination(corner_bl, bearing=b))[0:2]
    #         corner_tr = list(distance.distance(kilometers=S).destination(corner_br, bearing=-b))[0:2]
    #         dist_top = distance.distance(corner_tl, corner_tr).km

    # # finally, get individual locs
    # from_bottom = abs(coords[:,0] - corner_bl[0])
    # from_top = abs(coords[:,0] - corner_tl[0])
    # total_y = from_bottom + from_top
    # relative_y = (from_bottom / total_y)
    # longitudinal_stretch = abs(corner_bl[1]-corner_tl[1]) 
    # from_left = abs(coords[:,1] - (corner_bl[1]-(longitudinal_stretch*relative_y)))
    # from_right = abs(coords[:,1] - (corner_br[1]+(longitudinal_stretch*relative_y)))
    # total_x = from_left + from_right
    # relative_x = (from_left / total_x)
    # projection = [relative_x*S, relative_y*S]
    # projection = np.array(projection)
    # projection = projection.T
    # return projection
    ##########################################################################


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
    geno_mat, pos_list = [], []
    vcf = open(vcf_path, "r")
    current_chrom = "XX"
    baseline_pos = 0
    previous_pos = 0
    for line in vcf:
        if line[0] != "#":
            newline = line.strip().split("\t")
            genos = []
            for field in range(9, len(newline)):
                geno = newline[field].split(":")[0].split("/")
                geno = [int(geno[0]), int(geno[1])]
                if phase == 1:
                    genos.append(sum(geno)) 
                elif phase == 2:
                    geno = [min(geno), max(geno)]
                    genos.append(geno[0])
                    genos.append(geno[1])
                else:
                    print("problem")
                    exit()
            for i in range((max_n * phase) - len(genos)):  # pad with 0s
                genos.append(0)
            geno_mat.append(genos)

            # deal with genomic position
            chrom = newline[0]
            pos = int(newline[1])
            if chrom == current_chrom:
                previous_pos = int(pos)
            else:
                current_chrom = str(chrom)
                baseline_pos = (
                    int(previous_pos) + 10000
                )  # skipping 10kb between chroms/scaffolds (also skipping 10kb before first snp, currently)
            current_pos = baseline_pos + pos
            pos_list.append(current_pos)

    # check if enough snps
    if len(geno_mat) < num_snps:
        print("not enough snps")
        exit()
    if len(geno_mat[0]) < (max_n * phase):
        print("not enough samples")
        exit()

    # sample snps
    geno_mat = np.array(geno_mat)
    pos_list = np.array(pos_list)
    mask = [True] * num_snps + [False] * (geno_mat.shape[0] - num_snps)
    np.random.shuffle(mask)
    geno_mat = geno_mat[mask, :]
    pos_list = pos_list[mask]

    # rescale positions
    pos_list = pos_list / (max(pos_list) + 1)  # +1 to avoid prop=1.0

    return geno_mat, pos_list


### main
def main():
    vcf_path = sys.argv[1]
    max_n = int(sys.argv[2])
    num_snps = int(sys.argv[3])
    outname = sys.argv[4]
    phase = int(sys.argv[5])
    geno_mat, pos_list = vcf2genos(vcf_path, max_n, num_snps, phase)
    np.save(outname + ".genos", geno_mat)
    np.save(outname + ".pos", pos_list)


if __name__ == "__main__":
    main()
