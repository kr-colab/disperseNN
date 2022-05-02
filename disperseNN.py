import os
import argparse
from check_params import *
from read_input import *
from process_input import *
from data_generation import DataGenerator

def load_modules():
    print("loading bigger modules")
# *** need to check if these are all needed
    import sys, random, math, gc
    import argparse
    import numpy as np
    global tf
    import tensorflow as tf
    from tensorflow import keras
    import json
    global DataGenerator
    return

parser=argparse.ArgumentParser()
parser.add_argument("--train",action='store_true', default=False,help="run training pipeline")
parser.add_argument("--predict",action='store_true', default=False,help="run prediction pipeline")
parser.add_argument("--preprocessed",action='store_true', default=False,help="use preprocessed tensors, rather than tree sequences, as input")
parser.add_argument("--empirical",default=None,help="prefix for vcf and locs")
parser.add_argument("--target_list",help="list of PNG filepaths.",default=None)
parser.add_argument("--tree_list",help="list of tree filepaths.",default=None)
parser.add_argument("--width_list",help="list of map widths.", default=None)
parser.add_argument("--edge_width",help="crop a fixed edge width for each map",default=None,type=float)
parser.add_argument("--map_width",help="the whole habitat",default=None,type=float)
parser.add_argument("--sampling_width",help="just the sampling area",default=None,type=float)
parser.add_argument("--num_snps",default=None,type=int,help="maximum number of SNPs across all datasets (for pre-allocating memory)")
parser.add_argument("--num_pred",default=None,type=int,help="number of datasets to predict on")
parser.add_argument("--min_n",default=None,type=int,help="minimum number of samples (for pre-allocating memory)")
parser.add_argument("--max_n",default=None,type=int,help="maximum number of samples (for pre-allocating memory)")
parser.add_argument("--mu",help="mutation rate",default=1e-8,type=float)
parser.add_argument("--rho",help="recombination rate",default=1e-8,type=float)
parser.add_argument("--on_the_fly",default=1,type=int,help="number of samples (with replacement) from each tree sequence")
parser.add_argument("--validation_split",default=0.1,type=float,help="0-1, proportion of samples to use for validation. default: 0.1")
parser.add_argument("--batch_size",default=1,type=int, help="default: 10")
parser.add_argument("--bootstrap_reps_radseq",default=1,type=int, help="default: 1")
parser.add_argument("--max_epochs",default=100,type=int,help="default: 100")
parser.add_argument("--patience",type=int,default=100,help="n epochs to run the optimizer after last improvement in validation loss. default: 100")
parser.add_argument("--genome_length",default=1000000,type=int, help="important for rescaling the genomic positions.")
parser.add_argument("--dropout_prop",default=0,type=float, help="proportion of weights to zero at the dropout layer. \default: 0")
parser.add_argument('--recapitate',type=str, help="recapitate on-the-fly; True or False, no default")
parser.add_argument('--mutate',type=str, help="add mutations on-the-fly; True or False, no default")
parser.add_argument("--crop",default=None,type=float, help="map-crop size")
parser.add_argument("--out",help="file name stem for output",default=None)
parser.add_argument("--seed",default=None,type=int, help="random seed.")
parser.add_argument("--gpu_number",default=None,type=str)
parser.add_argument('--load_weights',default=None,type=str,help='Path to a _weights.hdf5 file to load weight from previous run.')
parser.add_argument('--load_model',default=None,type=str,help='Path to a _model.hdf5 file to load model from previous run.')
parser.add_argument('--phase',default=None,type=int,help='1 for unknown phase, 2 for known phase')
parser.add_argument('--polarize',default=None,type=int,help='2 for major/minor, 1 for ancestral/derived') 
parser.add_argument('--keras_verbose',default=1,type=int,
                    help='verbose argument passed to keras in model training. \
                    0 = silent. 1 = progress bars for minibatches. 2 = show epochs. \
                    Yes, 1 is more verbose than 2. Blame keras. \
                    default: 1. ')
parser.add_argument("--threads",default=1,type=int,help="num threads. default: all available CPU threads.")
parser.add_argument("--samplewidth_list",help="",default=None)
parser.add_argument("--geno_list",help="",default=None)
parser.add_argument("--pos_list",help="",default=None)
parser.add_argument("--loc_list",help="",default=None)
parser.add_argument("--training_mean",help="mean sigma from training",default=None,type=float)
parser.add_argument("--training_sd",help="sigma standard deviation from training",default=None,type=float)
parser.add_argument("--training_targets",help="map list for training data",default=None)
parser.add_argument("--learning_rate",default=1e-3,type=float,help="learning rate. Default=1e-3. 1e-4 seems to work well for some things.")
args=parser.parse_args()
check_params(args)





def load_network():
    # set seed, gpu                                       
    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
    if args.gpu_number is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number

    # update 1dconv+pool iterations based on number of SNPs                                               
    num_conv_iterations = int(len(str(args.num_snps)) - 3)

    ### 1d cnn                                                                                            
    conv_kernal_size = 2
    pooling_size = 10
    filter_size = 64

    geno_input = tf.keras.layers.Input(shape=(args.num_snps,args.max_n*args.phase))
    h = tf.keras.layers.Conv1D(filter_size, kernel_size=conv_kernal_size, activation='relu')(geno_input)
    h = tf.keras.layers.AveragePooling1D(pool_size=pooling_size)(h)
    for i in range(num_conv_iterations):
        filter_size += 44
        h = tf.keras.layers.Conv1D(filter_size, kernel_size=conv_kernal_size, activation='relu')(h)
        h = tf.keras.layers.AveragePooling1D(pool_size=pooling_size)(h)
    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(128,activation='relu')(h)

    position_input = tf.keras.layers.Input(shape=(args.num_snps,))
    m1 = tf.keras.layers.Dense(256,activation='relu')(position_input)
    h =  tf.keras.layers.concatenate([h,m1])
    h = tf.keras.layers.Dense(128,activation='relu')(h)

    loc_input = tf.keras.layers.Input(shape=(2,args.max_n))
    m2 = tf.keras.layers.Dense(64,name="m2_dense1")(loc_input)
    m2 = tf.keras.layers.Flatten()(m2)
    h =  tf.keras.layers.concatenate([h,m2])
    h = tf.keras.layers.Dense(128,activation='relu')(h)

    area_input = tf.keras.layers.Input(shape=(1))
    h =  tf.keras.layers.concatenate([h,area_input])
    h = tf.keras.layers.Dense(128,activation='relu')(h)

    h = tf.keras.layers.Dropout(args.dropout_prop)(h)
    output = tf.keras.layers.Dense(1,activation="linear")(h)
    model = tf.keras.Model(inputs=[geno_input,position_input,loc_input,area_input], outputs=[output])
    opt=tf.keras.optimizers.Adam(learning_rate=args.learning_rate) 
    model.compile(loss='mse', optimizer=opt)
    model.summary()

    # load weights
    if args.load_weights is not None:
        print("loading saved weights")
        model.load_weights(args.load_weights)
    else:
        if args.train == True and args.predict == True:
            weights=args.out+"_model.hdf5"
            print("loading weights:", weights)
            model.load_weights(weights)
        elif args.predict == True:
            print("where is the saved model?")
            exit()



    # callbacks
    checkpointer=tf.keras.callbacks.ModelCheckpoint(
                  filepath=args.out+"_model.hdf5",
                  verbose=args.keras_verbose,
                  save_best_only=True,
                  save_weights_only=False,
                  monitor="val_loss",
                  period=1)
    earlystop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                            min_delta=0,
                                            patience=args.patience)
    reducelr=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.5,
                                                  patience=int(args.patience/5),
                                               verbose=args.keras_verbose,
                                               mode='auto',
                                               min_delta=0,
                                               cooldown=0,
                                               min_lr=0)



    return model,checkpointer,earlystop,reducelr




def generator_params(targets,trees,widths,edges,shuffle,genos,poss,locs,sample_widths):
    params = {'targets': targets,
              'trees': trees,
              'num_snps': args.num_snps,
              'min_n': args.min_n,
              'max_n': args.max_n,
              'batch_size': args.batch_size,
              'genome_length':args.genome_length,
              'mu':args.mu,
              'threads':args.threads,
              'shuffle':shuffle,
              'rho':args.rho,
              'baseseed':args.seed,
              'recapitate':args.recapitate,
              'mutate':args.mutate,
              'crop':args.crop,
              'map_width':args.map_width,
              'widths':widths,
              'sampling_width':args.sampling_width,
              'edges':edges,
              'phase':args.phase,
              'polarize':args.polarize,
              'sample_widths':sample_widths,
              'genos':genos,
              'poss':poss,
              'locs':locs,
              'preprocessed':args.preprocessed
             }
    return params






def prep_trees_for_train():
    # read targets
    targets = read_single_value(args.target_list)
    total_sims = len(targets)
    edges = {}
    for i in range(total_sims):    
        if args.edge_width == None:
            edges[i] = float(targets[i]) # want raw sigma value for cropping edges away
        else:
            edges[i] = float(args.edge_width)
        targets[i] = np.log(targets[i])

    # normalize maps
    meanSig=np.mean(targets)
    sdSig=np.std(targets)
    targets= [(x-meanSig)/sdSig for x in targets] # center and scale                                                         
    target_dict = {}
    for i in range(total_sims): # attaching targets to training data                                               
        target_dict[i] = targets[i]

    # tree sequences
    trees = read_dict(args.tree_list)

    # map widths
    if args.width_list != None: # list of widths
        widths = read_single_value_dict(args.width_list)
    else:
        widths = {}
        for i in range(total_sims):
            widths[i] = float(args.map_width)        

    # split into pred,val,train sets    
    sim_ids = np.arange(0,total_sims)
    val=np.random.choice(sim_ids, round(args.validation_split*total_sims), replace=False)
    train=np.array([x for x in sim_ids if x not in val])

    # organize "partitions" to hand to data generator   
    partition = {}
    partition['train'] = []
    partition['validation'] = []
    num_reps = args.on_the_fly
    if (len(val)*num_reps)%args.batch_size!=0 or (len(train)*num_reps)%args.batch_size!=0:
        print("val/train sets each need to be divisible by batch_size; otherwise some batches will have missing data")
        exit()
    for i in train:
        for j in range(num_reps):
            partition['train'].append(i)
    for i in val:
        for j in range(num_reps):
            partition['validation'].append(i)

    # initialize generators
    params = generator_params(target_dict,trees,widths,edges,True,None,None,None,None)
    training_generator = DataGenerator(partition['train'], **params) 
    validation_generator = DataGenerator(partition['validation'], **params)

    # train
    load_modules()
    model,checkpointer,earlystop,reducelr = load_network()
    print("training!")
    history = model.fit_generator(generator=training_generator,
                                  use_multiprocessing=False,
                                  epochs=args.max_epochs,
                                  shuffle=False, # (redundant with shuffling inside the generator)
                                  verbose=args.keras_verbose,
                                  validation_data=validation_generator,
                                  callbacks=[checkpointer,earlystop,reducelr])
    return 





def prep_preprocessed_for_train():
    # read targets                
    print("loading input data; this could take a while if the lists are long")
    targets = read_single_value(args.target_list)
    total_sims = len(targets)
    edges = {}
    for i in range(total_sims):
        if args.edge_width == None:
            edges[i] = float(targets[i]) # want raw sigma value for cropping edges away  
        else:
            edges[i] = float(args.edge_width)
        targets[i] = np.log(targets[i])

    # normalize maps                                                                     
    meanSig=np.mean(targets)
    sdSig=np.std(targets)
    targets= [(x-meanSig)/sdSig for x in targets] # center and scale                     
    target_dict = {}
    for i in range(total_sims): # attaching targets to training data                     
        target_dict[i] = targets[i]

    # other inputs
    sample_widths = load_single_value_dict(args.samplewidth_list)
    genos = read_dict(args.geno_list)
    locs = read_dict(args.loc_list)
    pos = read_dict(args.pos_list)
    
    # split into pred,val,train sets                                                     
    sim_ids = np.arange(0,total_sims)
    val=np.random.choice(sim_ids, round(args.validation_split*total_sims), replace=False)
    train=np.array([x for x in sim_ids if x not in val])

    # organize "partitions" to hand to data generator                                                                 
    partition = {}
    partition['train'] = list(train)
    partition['validation'] = list(val)

    # initialize generators                                                                                           
    params = generator_params(target_dict,None,None,None,True,genos,pos,locs,sample_widths)
    training_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['validation'], **params)

    # train                                                                                                           
    load_modules()
    model,checkpointer,earlystop,reducelr = load_network()
    print("training!")
    history = model.fit_generator(generator=training_generator,
                                  use_multiprocessing=False,
                                  epochs=args.max_epochs,
                                  shuffle=False,
                                  verbose=args.keras_verbose,
                                  validation_data=validation_generator,
                                  callbacks=[checkpointer,earlystop,reducelr])
    return





def prep_empirical(meanSig,sdSig):
    # convert locs                                                                                        
    locs = read_locs(args.empirical+".locs")
    locs = project_locs(locs,args.out,args.seed)
    locs,sampling_width=rescale_locs(locs)
    locs = pad_locs(locs,args.max_n)
    locs = np.reshape(locs, (1,locs.shape[0],locs.shape[1]))
    sampling_width = np.reshape(sampling_width, (1))

    # load model
    load_modules()
    model,checkpointer,earlystop,reducelr = load_network()

    # convert vcf to geno matrix                                                                          
    for i in range(args.bootstrap_reps_radseq):
        test_genos,test_pos= vcf2genos(args.empirical+".vcf",args.max_n,args.num_snps,args.phase)
        test_genos = np.reshape(test_genos, (1,test_genos.shape[0], test_genos.shape[1]))
        test_pos = np.reshape(test_pos, (1,test_pos.shape[0]))
        dataset=args.empirical + "_" + str(i) 
        prediction = model.predict( [test_genos,test_pos,locs,sampling_width] )
        unpack_predictions(prediction,meanSig,sdSig,None,dataset)

    return





def prep_empirical_preprocessed(meanSig,sdSig):

    # load inputs
    print("reading inputs: this can take a while if the lists are long")
    genos = read_dict(args.geno_list)
    datasets = np.array(read_list(args.geno_list)) # (filenames for output)
    poss = read_dict(args.pos_list)
    locs = read_locs(args.empirical+".locs")
    locs = project_locs(locs,args.out,args.seed)
    locs,sampling_width=rescale_locs(locs)
    locs =pad_locs(locs,args.max_n)
    locs = np.reshape(locs, (1,locs.shape[0],locs.shape[1]))
    sampling_width = np.reshape(sampling_width, (1))

    # load model
    load_modules()
    model,checkpointer,earlystop,reducelr = load_network()

    #predict
    print("predicting")
    for i in range(len(datasets)):
        test_genos = np.load(genos[i])
        test_pos = np.load(poss[i])
        test_genos = np.reshape(test_genos, (1,test_genos.shape[0], test_genos.shape[1]))
        test_pos = np.reshape(test_pos, (1,test_pos.shape[0]))
        prediction = model.predict( [test_genos,test_pos,locs,sampling_width] )
        unpack_predictions(prediction,meanSig,sdSig,None,datasets[i])

    return



def prep_preprocessed_for_pred(meanSig,sdSig):
    # load inputs                                                                          
    print("reading inputs: this can take a while if the lists are long")
    genos = read_dict(args.geno_list)
    poss = read_dict(args.pos_list)
    locs = read_dict(args.loc_list)
    sample_widths = load_single_value_dict(args.samplewidth_list)
    targets = read_single_value_dict(args.target_list)
    for i in range(len(targets)):
        targets[i] = np.log(targets[i])

    # organize "partition" to hand to data generator                                       
    partition = {}
    simids = np.random.choice(np.arange(0,len(genos)), args.num_pred, replace=False)
    partition['prediction'] = simids

    # get generator ready                                                                  
    params = generator_params(targets,None,None,None,False,genos,poss,locs,sample_widths)
    generator = DataGenerator(partition['prediction'], **params)

    # predict                                                                              
    load_modules()
    model,checkpointer,earlystop,reducelr = load_network()
    print("predicting")
    predictions=model.predict_generator(generator)
    unpack_predictions(predictions,meanSig,sdSig,targets,simids)

    return






def prep_trees_for_pred(meanSig,sdSig):
    # read inputs (note these are lists instead of dict.)
    trees = read_list(args.tree_list) 
    targets = read_single_value(args.target_list)
    total_sims = len(targets)
    edges = list(targets) # setting edge width to sigma
    targets = np.log(targets) 
    if args.width_list != None: # list of widths         
        widths = read_single_value(args.width_list)
    else:
        widths = []
        for i in range(total_sims):
            widths.append(args.map_width)

    # organize "partition" to hand to data generator                                                                                     
    partition = {}
    if (args.num_pred)%args.batch_size!=0:
        print("\n\npred sets each need to be divisible by batch_size; otherwise some batches will have missing data\n\n")
        exit()
    partition['prediction'] = np.arange(0,args.num_pred)

    # get generator ready
    params = generator_params(targets,trees,widths,edges,False,None,None,None,None) 
    generator = DataGenerator(partition['prediction'], **params)
    
    # predict
    load_modules()
    model,checkpointer,earlystop,reducelr = load_network()
    print("predicting")
    predictions=model.predict_generator(generator)
    unpack_predictions(predictions,meanSig,sdSig,targets,trees)

    return




def unpack_predictions(predictions,meanSig,sdSig,targets,datasets):
    squared_errors = []
        
    if args.empirical == None:
        for i in range(len(predictions)):
            if args.preprocessed == False:
                trueval = targets[i]         
            else:
                trueval = targets[datasets[i]]
            prediction = (predictions[i][0] * sdSig) + meanSig
            error = (trueval-prediction)**2
            squared_errors.append(error)
            print(datasets[i], np.round(trueval,10), np.round(prediction,10))
        print("RMSE:", np.mean(squared_errors)**(1/2))

    else:
        prediction = predictions[0]
        prediction = (prediction * sdSig) + meanSig
        prediction = np.exp(prediction)
        prediction = np.round(prediction,10)
        print(datasets, prediction, "(exponentiated)")

    return
    






### main ###

# train
if args.train == True:
    print("starting training pipeline")
    # prep input and initialize generators
    if args.preprocessed == False:
        print("using tree sequences")
        prep_trees_for_train()
    else:        
        print("using pre-processed tensors")
        prep_preprocessed_for_train()

# predict        
if args.predict == True:
    print("starting prediction pipeline")
    # first we need the mean and sd from training to back-transform the z-normalization
    if args.training_targets != None:
        train_targets = read_single_value(args.training_targets)
        train_targets = list(map(float,train_targets))
        train_targets = np.log(train_targets)
        meanSig = np.mean(train_targets)
        sdSig = np.std(train_targets)
    elif args.training_mean != None:
        meanSig,sdSig = args.training_mean, args.training_sd
    print("mean and sd:",meanSig,sdSig)

    # prep_input
    if args.empirical == None:
        print("predicting on simulated data")
        if args.preprocessed == True:
            print("using pre-processed tensors")
            prep_preprocessed_for_pred(meanSig,sdSig)
        else:
            print("using tree sequences")
            prep_trees_for_pred(meanSig,sdSig)
    else:
        print("predicting on empirical data")
        if args.preprocessed == True:
            print("using pre-processed tensors")
            prep_empirical_preprocessed(meanSig,sdSig)
        else:
            print("using a single VCF")
            prep_empirical(meanSig,sdSig)





