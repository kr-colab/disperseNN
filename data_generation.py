# data generator code for training CNN

import sys
import numpy as np
import tensorflow as tf
import msprime
import tskit
import math
import pyslim
import time
import multiprocessing
import gc
import warnings

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, targets, trees, genome_length, threads, batch_size, num_snps, min_n, max_n, mu, rho, baseseed, recapitate, mutate, crop, map_width, widths, sampling_width, edges, phase, polarize, sample_widths, genos, poss, locs, preprocessed, shuffle=True):
        'Initialization'
        self.num_snps = num_snps
        self.min_n = min_n
        self.max_n = max_n
        self.batch_size = batch_size
        self.targets = targets
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.trees = trees
        self.genome_length = genome_length
        self.mu = mu
        self.threads = threads
        self.rho = rho
        self.baseseed = baseseed
        self.recapitate = recapitate
        self.mutate = mutate
        self.crop = crop
        self.map_width = map_width
        self.widths = widths
        self.sampling_width = sampling_width
        self.edges = edges
        self.phase = phase
        self.polarize = polarize
        self.sample_widths = sample_widths
        self.genos = genos
        self.poss = poss
        self.locs = locs
        self.preprocessed = preprocessed
        self.on_epoch_end()
        np.random.seed(self.baseseed)
        warnings.simplefilter('ignore', msprime.TimeUnitsMismatchWarning)
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def cropper(self,ts,W,sample_width,edge_width,alive_inds):
        cropped=[]
        left_edge = np.random.uniform(low=edge_width, high=W-edge_width-sample_width)
        right_edge = left_edge+sample_width
        bottom_edge = np.random.uniform(low=edge_width, high=W-edge_width-sample_width)
        top_edge = bottom_edge+sample_width
        for i in alive_inds:
            ind = ts.individual(i)
            loc = ind.location[0:2]
            if loc[0] > left_edge and loc[0] < right_edge and loc[1] > bottom_edge and loc[1] < top_edge:
                cropped.append(i)
        return cropped



    def sample_ts(self, filepath, W, edge_width, seed):
        sys.stderr.flush()

        # read input
        #print(filepath)
        sys.stdout.flush()
        ts = tskit.load(filepath)
        np.random.seed(seed)

        # recapitate
        alive_inds = []
        for i in ts.individuals():
            alive_inds.append(i.id)
        if self.recapitate == "True":
            N = len(alive_inds)
            #ts = ts.recapitate(recombination_rate=self.rho, Ne=N, random_seed=seed) 
            ts = pyslim.recapitate(ts,
                                   recombination_rate=self.rho,
                                   ancestral_Ne=N,
                                   random_seed=seed) 
            
        # crop map
        if self.sampling_width != None:
            sample_width = (float(self.sampling_width)*W)-(edge_width*2)
        else:
            sample_width = np.random.uniform(0,W-(edge_width*2)) # maybe change to log-U once you have plenty of big maps, W>100
            ### for Job145 only
            #sample_width = np.random.uniform(0,40) 
            #sample_width = np.random.uniform(40,W-(edge_width*2))
            ###
        n = np.random.randint(self.min_n,self.max_n+1) # (excludes the max of the range)
        sampled_inds = self.cropper(ts,W,sample_width,edge_width,alive_inds)
        failsafe = 0
        while len(sampled_inds) < n: # keep looping until you get a map with enough samples
            if self.sampling_width != None:
                sample_width = (float(self.sampling_width)*W)-(edge_width*2)
            else:
                sample_width = np.random.uniform(0,W-(edge_width*2))
                ### for Job145 only                   
                #sample_width = np.random.uniform(0,40)
                #sample_width = np.random.uniform(40,W-(edge_width*2))
                ###                                   
            n = np.random.randint(self.min_n,self.max_n+1) # (excludes the max of the range)
            failsafe += 1
            if failsafe > 100:
                print("\tnot enough samples, killed while-loop after 100 loops")
                sys.stdout.flush()
                exit()
            sampled_inds = self.cropper(ts,W,sample_width,edge_width,alive_inds)

        # sample individuals
        keep_indivs = np.random.choice(sampled_inds, n, replace=False)
        keep_nodes = []                 
        for i in keep_indivs:           
            ind = ts.individual(i)      
            keep_nodes.extend(ind.nodes)

        # simplify
        ts = ts.simplify(keep_nodes) 

        # mutate
        if self.mutate == "True":
            mu=float(self.mu)
            ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed, model=msprime.SLiMMutationModel(type=0), keep=True)
            counter = 0
            while ts.num_sites < (self.num_snps*self.polarize):
                sys.stdout.flush()
                counter += 1
                mu*=10
                ts = msprime.sim_mutations(ts, rate=mu,random_seed=seed, model=msprime.SLiMMutationModel(type=0), keep=True)
                if counter == 10:
                    print("\n\nsorry, Dude. Didn't generate enough snps. \n\n")
                    sys.stdout.flush()
                    exit()

        # grab spatial locations
        sample_dict = {}
        locs0 = []
        for samp in ts.samples():
            node = ts.node(samp)
            indID = node.individual
            if indID not in sample_dict:
                sample_dict[indID] = 0
                loc = ts.individual(indID).location[0:2]
                locs0.append(loc)

        # rescale locs to (0,1) 
        locs0=np.array(locs0) 
        minx = min(locs0[:,0])
        maxx = max(locs0[:,0])
        miny = min(locs0[:,1])
        maxy = max(locs0[:,1])
        x_range = maxx - minx
        y_range = maxy - miny
        sample_width = max(x_range,y_range) # re-define width to be this distance
        locs0[:,0] =(locs0[:,0] - minx) / x_range # rescale to (0,1)
        locs0[:,1] =(locs0[:,1] - miny) / y_range
        if x_range > y_range: # these four lines for preserving aspect ratio    
            locs0[:,1] *= (y_range / x_range) 
        elif y_range > x_range:
            locs0[:,0] *= (x_range / y_range) 
            
        # stuff locs into sparse array
        locs = np.zeros((self.max_n,2))
        locs[0:n, :] = locs0
        locs=locs.T

        # grab genos and positions                    
        geno_mat0 = ts.genotype_matrix()
        pos_list = ts.tables.sites.position

        # change 0,1 encoding to major/minor allele                                                                    
        if self.polarize == 2:
            mask = [True]*(self.num_snps*2) + [False]*(ts.num_sites - (self.num_snps*2)) # (*2 is arbitrary)
            np.random.shuffle(mask)
            geno_mat0,pos_list = geno_mat0[mask,:],pos_list[mask] # (keeping these together to keep things simple)
            geno_mat1 = []
            pos_list1 = []
            for i in range(geno_mat0.shape[0]):
                alleles = {}
                for j in range(n*2): # diploid at this point, even if you want to collapse genotypes (which comes next)
                    a = geno_mat0[i][j]
                    if a not in alleles:
                        alleles[a] = 0
                    alleles[a] += 1
                if len(alleles) == 2:
                    new_genotypes = []
                    major,minor=list(set(alleles)) # set() gives random order                                              
                    if alleles[major] < alleles[minor]:
                        major,minor=minor,major
                    for j in range(n*2): # go back through and convert genotypes                                           
                        a = geno_mat0[i][j]
                        if a == major:
                            new_genotype = 0
                        elif a == minor:
                            new_genotype = 1
                        new_genotypes.append(new_genotype)
                    geno_mat1.append(new_genotypes)
                    pos_list1.append(pos_list[i])
            geno_mat0 = np.array(geno_mat1)
            pos_list = np.array(pos_list1)

        # sample SNPs                                                       
        mask = [True]*self.num_snps + [False]*(geno_mat0.shape[0] - self.num_snps)
        np.random.shuffle(mask)
        geno_mat0 = geno_mat0[mask,:]
        pos_list = pos_list[mask]

        # unphase the genotypes, change to allele dosage (e.g. 0,1,2)
        if self.phase == 1:
            geno_mat1 = np.zeros((self.num_snps,n))
            for ind in range(n):
                geno_mat1[:,ind] += geno_mat0[:,ind*2]
                geno_mat1[:,ind] += geno_mat0[:,ind*2+1]
            geno_mat0 = np.array(geno_mat1)
        
        # shove the retained snps into padded genotype matrix
        geno_mat = np.zeros((self.num_snps, self.max_n*self.phase))
        geno_mat[:,0:n*self.phase] = geno_mat0

        # rescale genomic positions by genome length
        pos_list = pos_list / self.genome_length

        del ts
        del geno_mat0
        #del geno_mat
        #del pos_list
        del mask
        del alive_inds
        del sampled_inds
        gc.collect()

        sys.stdout.flush()
        sys.stderr.flush()

        return geno_mat,pos_list,locs,sample_width
    

    def preprocess_sample_ts(self, geno_path, pos_path, loc_path):

        # read input                                              
        geno_mat = np.load(geno_path)
        pos_list = np.load(pos_path)
        locs = np.load(loc_path)

        return geno_mat,pos_list,locs

            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        sys.stderr.flush()

        # Initialization
        X1 = np.empty( (self.batch_size, self.num_snps, self.max_n*self.phase), dtype="int8" ) # vcf
        X2 = np.empty( (self.batch_size, self.num_snps) )# positions 
        X3 = np.empty( (self.batch_size, 2, self.max_n) ) # locs
        X4 = np.empty( (self.batch_size,) ) # sample widths
        y = np.empty( (self.batch_size), dtype=float) # targets (sigma)

        if self.preprocessed == False:
            ts_list = []
            width_list = []
            edge_list = []
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.targets[ID]
                filepath = self.trees[ID]
                ts_list.append(filepath)
                if self.map_width != None:
                    width_list.append(self.map_width)
                else:
                    width_list.append(self.widths[ID])
                edge_list.append(self.edges[ID])
            sys.stdout.flush()
            seeds = np.random.randint(1e9, size=(self.batch_size))

            sys.stdout.flush()
            pool = multiprocessing.Pool(self.threads, maxtasksperchild=1) 
            batch = pool.starmap(self.sample_ts, zip(ts_list, width_list, edge_list, seeds))

            # unpack the multiprocess output
            for i in range(len(batch)): 
                X1[i,:] = batch[i][0]
                X2[i,:] = batch[i][1]
                X3[i,:] = batch[i][2]
                X4[i]   = batch[i][3]
            X = [X1,X2,X3,X4]

        else:
            geno_list = []
            pos_list = []
            loc_list = []
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.targets[ID]
                X4[i] = self.sample_widths[ID]
                loc_list.append(self.locs[ID])
                geno_list.append(self.genos[ID])
                pos_list.append(self.poss[ID])
            sys.stdout.flush()
            seeds = np.random.randint(1e9, size=(self.batch_size))
            sys.stdout.flush()
            pool = multiprocessing.Pool(self.threads, maxtasksperchild=1)
            batch = pool.starmap(self.preprocess_sample_ts, zip(geno_list, pos_list, loc_list))
            # unpack the multiprocess output                                                   
            for i in range(len(batch)):
                X1[i,:] = batch[i][0]
                X2[i,:] = batch[i][1]
                X3[i,:] = batch[i][2][0]
            X = [X1,X2,X3,X4]

        gc.collect()
        sys.stdout.flush()
        sys.stderr.flush()


        return (X, y)
