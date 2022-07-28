# main code for disperseNN

import os
import argparse
import tskit
from sklearn.model_selection import train_test_split
from check_params import *
from read_input import *
from process_input import *
from data_generation import DataGenerator

def load_dl_modules():
    print("loading bigger modules")
    import numpy as np
    global tf
    import tensorflow as tf
    from tensorflow import keras
    return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train", action="store_true", default=False, help="run training pipeline"
)
parser.add_argument(
    "--predict", action="store_true", default=False, help="run prediction pipeline"
)
parser.add_argument(
    "--preprocessed",
    action="store_true",
    default=False,
    help="use preprocessed tensors, rather than tree sequences, as input",
)
parser.add_argument("--empirical", default=None, help="prefix for vcf and locs")
parser.add_argument("--target_list", help="list of PNG filepaths.", default=None)
parser.add_argument("--tree_list", help="list of tree filepaths.", default=None)
parser.add_argument(
    "--edge_width",
    help="crop a fixed width from each edge of the map; enter 'sigma' to set edge_width equal to sigma ",
    default=None,
    type=str,
)
parser.add_argument(
    "--sampling_width", help="just the sampling area", default=None, type=float
)
parser.add_argument(
    "--num_snps",
    default=None,
    type=int,
    help="maximum number of SNPs across all datasets (for pre-allocating memory)",
)
parser.add_argument(
    "--num_pred", default=None, type=int, help="number of datasets to predict on"
)
parser.add_argument(
    "--min_n",
    default=None,
    type=int,
    help="minimum sample size",
)
parser.add_argument(
    "--max_n",
    default=None,
    type=int,
    help="maximum sample size",
)
parser.add_argument(
    "--mu",
    help="beginning mutation rate: mu is increased until num_snps is achieved",
    default=1e-15,
    type=float,
)
parser.add_argument("--rho", help="recombination rate", default=1e-8, type=float)
parser.add_argument(
    "--num_samples",
    default=1,
    type=int,
    help="number of organismal-samples (each of size n) from each tree sequence",
)
parser.add_argument(
    "--num_reps",
    default=1,
    type=int,
    help="number of replicate-draws from the genotype matrix of each sample",
)
parser.add_argument(
    "--validation_split",
    default=0.2,
    type=float,
    help="0-1, proportion of samples to use for validation. default: 0.2",
)
parser.add_argument("--batch_size", default=1, type=int, help="default: 1")
parser.add_argument("--max_epochs", default=1000, type=int, help="default: 100")
parser.add_argument(
    "--patience",
    type=int,
    default=100,
    help="n epochs to run the optimizer after last improvement in validation loss.",
)
parser.add_argument(
    "--dropout",
    default=0,
    type=float,
    help="proportion of weights to zero at the dropout layer. \default: 0",
)
parser.add_argument(
    "--recapitate", type=str, help="recapitate on-the-fly; True or False"
)
parser.add_argument(
    "--mutate", type=str, help="add mutations on-the-fly; True or False"
)
parser.add_argument("--crop", default=None, type=float, help="map-crop size")
parser.add_argument(
    "--out", help="file name stem for output", default=None, required=True
)
parser.add_argument("--seed", default=None, type=int, help="random seed.")
parser.add_argument("--gpu_index", default="-1", type=str, help="index of gpu. To avoid GPUs, skip this flag or say '-1'. To use any available GPU say 'any' ")
parser.add_argument(
    "--load_weights",
    default=None,
    type=str,
    help="Path to a _weights.hdf5 file to load weight from previous run.",
)
parser.add_argument(
    "--load_model",
    default=None,
    type=str,
    help="Path to a _model.hdf5 file to load model from previous run.",
)
parser.add_argument(
    "--phase",
    default=1,
    type=int,
    help="1 for unknown phase, 2 for known phase",
)
parser.add_argument(
    "--polarize",
    default=2,
    type=int,
    help="2 for major/minor, 1 for ancestral/derived",
)
parser.add_argument(
    "--keras_verbose",
    default=1,
    type=int,
    help="verbose argument passed to keras in model training. \
                    0 = silent. 1 = progress bars for minibatches. 2 = show epochs. \
                    Yes, 1 is more verbose than 2. Blame keras. \
                    default: 1. ",
)
parser.add_argument(
    "--threads",
    default=1,
    type=int,
    help="num threads. default: all available CPU threads.",
)
parser.add_argument("--samplewidth_list", help="", default=None)
parser.add_argument("--geno_list", help="", default=None)
parser.add_argument(
    "--training_params", help="params used in training: sigma mean and sd, max_n, num_snps", default=None
)
parser.add_argument(
    "--learning_rate",
    default=1e-3,
    type=float,
    help="learning rate.",
)
args = parser.parse_args()
check_params(args)


def load_network():
    # set seed, gpu
    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
    if args.gpu_index != 'any': # 'any' will search for any available GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # update conv+pool iterations based on number of SNPs
    num_conv_iterations = int(np.floor(np.log10(args.num_snps))-2)
    if num_conv_iterations < 0:
        num_conv_iterations = 0

    # cnn architecture
    conv_kernal_size = 2
    pooling_size = 10
    filter_size = 64

    geno_input = tf.keras.layers.Input(shape=(args.num_snps, args.max_n * args.phase))
    h = tf.keras.layers.Conv1D(
        filter_size, kernel_size=conv_kernal_size, activation="relu"
    )(geno_input)
    h = tf.keras.layers.AveragePooling1D(pool_size=pooling_size)(h)
    for i in range(num_conv_iterations):
        filter_size += 44
        h = tf.keras.layers.Conv1D(
            filter_size, kernel_size=conv_kernal_size, activation="relu"
        )(h)
        h = tf.keras.layers.AveragePooling1D(pool_size=pooling_size)(h)

    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(128, activation="relu")(h)
    h = tf.keras.layers.Dense(128, activation="relu")(h)
    h = tf.keras.layers.Dense(128, activation="relu")(h)

    width_input = tf.keras.layers.Input(shape=(1))
    h = tf.keras.layers.concatenate([h, width_input])
    h = tf.keras.layers.Dense(128, activation="relu")(h)

    h = tf.keras.layers.Dropout(args.dropout)(h)
    output = tf.keras.layers.Dense(1, activation="linear")(h)
    model = tf.keras.Model(
        inputs=[geno_input, width_input], outputs=[output]
    )
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(loss="mse", optimizer=opt)
    model.summary()

    # load weights
    if args.load_weights is not None:
        print("loading saved weights")
        model.load_weights(args.load_weights)
    else:
        if args.train == True and args.predict == True:
            weights = args.out + "_model.hdf5"
            print("loading weights:", weights)
            model.load_weights(weights)
        elif args.predict == True:
            print("where is the saved model? (via --load_weights)")
            exit()

    # callbacks
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.out + "_model.hdf5",
        verbose=args.keras_verbose,
        save_best_only=True,
        save_weights_only=False,
        monitor="val_loss",
        period=1,
    )
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=args.patience
    )
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=int(args.patience/10),
        verbose=args.keras_verbose,
        mode="auto",
        min_delta=0,
        cooldown=0,
        min_lr=0,
    )

    return model, checkpointer, earlystop, reducelr


def make_generator_params_dict(
    targets, trees, shuffle, genos, sample_widths
):
    params = {
        "targets": targets,
        "trees": trees,
        "num_snps": args.num_snps,
        "min_n": args.min_n,
        "max_n": args.max_n,
        "batch_size": args.batch_size,
        "mu": args.mu,
        "threads": args.threads,
        "shuffle": shuffle,
        "rho": args.rho,
        "baseseed": args.seed,
        "recapitate": args.recapitate,
        "mutate": args.mutate,
        "crop": args.crop,
        "sampling_width": args.sampling_width,
        "edge_width": args.edge_width,
        "phase": args.phase,
        "polarize": args.polarize,
        "sample_widths": sample_widths,
        "genos": genos,
        "preprocessed": args.preprocessed,
        "num_reps": args.num_reps
    }
    return params


def prep_trees_and_train():

    # tree sequences                 
    trees = read_dict(args.tree_list)
    total_sims = len(trees)

    # read targets
    print("reading targets from tree sequences: this should take several minutes")
    targets = []
    for i in range(total_sims):
        ts = tskit.load(trees[i]) ### *** can we access provenance without loading the whole tree sequence?
        target = parse_provenance(ts, 'sigma')
        target = np.log(target)
        targets.append(target)
        
    # normalize targets
    meanSig = np.mean(targets)
    sdSig = np.std(targets)
    np.save(f"{args.out}_training_params", [meanSig,sdSig,args.max_n,args.num_snps])    
    targets = [(x - meanSig) / sdSig for x in targets]  # center and scale
    targets = dict_from_list(targets)

    # split into val,train sets
    sim_ids = np.arange(0, total_sims)
    train, val = train_test_split(sim_ids, test_size=args.validation_split)    
    if len(val)*args.num_samples % args.batch_size != 0 or len(train)*args.num_samples % args.batch_size != 0:
        print(
            "\n\ntrain and val sets each need to be divisible by batch_size; otherwise some batches will have missing data\n\n"
        )
        exit()
    
    # organize "partitions" to hand to data generator
    partition = {}
    partition["train"] = []
    partition["validation"] = []
    for i in train:
        for j in range(args.num_samples):
            partition["train"].append(i)
    for i in val:
        for j in range(args.num_samples):
            partition["validation"].append(i)

    # initialize generators
    params = make_generator_params_dict(
        targets=targets,
        trees=trees,
        shuffle=True,
        genos=None,
        sample_widths=None,
    )
    training_generator = DataGenerator(partition["train"], **params)
    validation_generator = DataGenerator(partition["validation"], **params)

    # train
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network()
    print("training!")
    history = model.fit_generator(
        generator=training_generator,
        use_multiprocessing=False,
        epochs=args.max_epochs,
        shuffle=False,  # (redundant with shuffling inside the generator)
        verbose=args.keras_verbose,
        validation_data=validation_generator,
        callbacks=[checkpointer, earlystop, reducelr],
    )

    return


def prep_preprocessed_and_train():
    
    # read targets
    print("loading input data; this could take a while if the lists are very long")
    print("\ttargets")
    sys.stderr.flush()
    targets = read_single_value(args.target_list)
    targets = np.log(targets)
    total_sims = len(targets)

    # normalize targets                                                              
    meanSig = np.mean(targets)
    sdSig = np.std(targets)
    np.save(f"{args.out}_training_params", [meanSig,sdSig,args.max_n,args.num_snps])    
    targets = [(x - meanSig) / sdSig for x in targets]  # center and scale
    targets = dict_from_list(targets)
        
    # other inputs
    print("\tsample_widths")
    sys.stderr.flush()
    sample_widths = load_single_value_dict(args.samplewidth_list)
    print("\tgenos")
    sys.stderr.flush()
    genos = read_dict(args.geno_list)

    # split into val,train sets
    sim_ids = np.arange(0, total_sims)
    train, val = train_test_split(sim_ids, test_size=args.validation_split)
    if len(val)*args.num_samples % args.batch_size != 0 or len(train)*args.num_samples % args.batch_size != 0:
        print(
            "\n\ntrain and val sets each need to be divisible by batch_size; otherwise some batches will have missing data\n\n"
        )
        exit()

    # organize "partitions" to hand to data generator
    partition = {}
    partition["train"] = list(train)
    partition["validation"] = list(val)

    # initialize generators
    params = make_generator_params_dict(
        targets=targets,
        trees=None,
        shuffle=True,
        genos=genos,
        sample_widths=sample_widths,
    )
    training_generator = DataGenerator(partition["train"], **params)
    validation_generator = DataGenerator(partition["validation"], **params)

    # train
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network()
    print("training!")
    history = model.fit_generator(
        generator=training_generator,
        use_multiprocessing=False,
        epochs=args.max_epochs,
        shuffle=False,
        verbose=args.keras_verbose,
        validation_data=validation_generator,
        callbacks=[checkpointer, earlystop, reducelr],
    )

    return


def prep_empirical_and_pred():

    # grab mean and sd from training distribution                                 
    meanSig,sdSig,args.max_n,args.num_snps = np.load(args.training_params)
    args.max_n = int(args.max_n)
    args.num_snps = int(args.num_snps)

    # project locs
    locs = read_locs(args.empirical + ".locs")
    locs = np.array(locs)
    sampling_width = project_locs(locs)
    print("sampling_width:", sampling_width)
    sampling_width = np.reshape(sampling_width, (1))

    # load model
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network()

    # convert vcf to geno matrix
    for i in range(args.num_reps):
        test_genos = vcf2genos(
            args.empirical + ".vcf", args.max_n, args.num_snps, args.phase
        )
        ibd(test_genos, locs, args.phase, args.num_snps)
        test_genos = np.reshape(
            test_genos, (1, test_genos.shape[0], test_genos.shape[1])
        )
        dataset = args.empirical + "_" + str(i)
        prediction = model.predict([test_genos, sampling_width])
        unpack_predictions(prediction, meanSig, sdSig, None, dataset, dataset) 

    return


def prep_preprocessed_and_pred():
   
    # grab mean and sd from training distribution                                 
    meanSig,sdSig,args.max_n,args.num_snps = np.load(args.training_params)
    args.max_n = int(args.max_n)
    args.num_snps = int(args.num_snps)

    # load inputs
    targets = read_single_value(args.target_list)
    targets = np.log(targets)
    targets = dict_from_list(targets)
    genos = read_dict(args.geno_list)
    sample_widths = load_single_value_dict(args.samplewidth_list)
    file_names = read_dict(args.geno_list) # storing just the names, for output.

    # organize "partition" to hand to data generator
    partition = {}
    simids = np.random.choice(np.arange(0, len(genos)), args.num_pred, replace=False)
    partition["prediction"] = simids

    # get generator ready
    params = make_generator_params_dict(
        targets=targets,
        trees=None,
        shuffle=False,
        genos=genos,
        sample_widths=sample_widths,
    )
    generator = DataGenerator(partition["prediction"], **params)

    # predict
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network()
    print("predicting")
    predictions = model.predict_generator(generator)
    unpack_predictions(predictions, meanSig, sdSig, targets, simids, file_names)

    return


def prep_trees_and_pred():

    # grab mean and sd from training distribution                                 
    meanSig,sdSig,args.max_n,args.num_snps = np.load(args.training_params)
    args.max_n = int(args.max_n)
    args.num_snps = int(args.num_snps)

    # tree sequences                                                
    trees = read_dict(args.tree_list)
    total_sims = len(trees)

    # read targets                                                                
    print("reading true values from tree sequences: this should take several minutes")
    targets = []
    for i in range(total_sims):
        ts = tskit.load(trees[i])
        target = parse_provenance(ts, 'sigma')
        target = np.log(target)
        targets.append(target)

    # organize "partition" to hand to data generator
    partition = {}
    simids = np.random.choice(np.arange(0, total_sims), args.num_pred, replace=False)
    partition["prediction"] = simids

    # get generator ready
    params = make_generator_params_dict(
        targets=[None]*total_sims, 
        trees=trees,
        shuffle=False,
        genos=None,
        sample_widths=None,
    )
    generator = DataGenerator(partition["prediction"], **params)
    
    # predict
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network()
    print("predicting")
    predictions = model.predict_generator(generator)
    unpack_predictions(predictions, meanSig, sdSig, targets, simids, trees)

    return


def unpack_predictions(predictions, meanSig, sdSig, targets, simids, file_names):
    if args.empirical == None:
        with open(f"{args.out}_predictions.txt", "w") as out_f:
            raes = []
            for i in range(args.num_pred):
                for r in range(args.num_reps):
                    pred_index = r + (i*args.num_reps)
                    trueval = float(targets[simids[i]])
                    prediction = predictions[pred_index][0]
                    prediction = (prediction * sdSig) + meanSig
                    trueval = np.exp(trueval)
                    prediction = np.exp(prediction)
                    rae = abs( (trueval - prediction) / trueval )
                    raes.append(rae)
                    print(file_names[i], np.round(trueval, 10), np.round(prediction, 10), file=out_f)
        print("mean RAE:", np.mean(raes))
    else:
        with open(f"{args.out}_predictions.txt", "a") as out_f:
            prediction = predictions[0][0]
            prediction = (prediction * sdSig) + meanSig
            prediction = np.exp(prediction)
            prediction = np.round(prediction, 10)
            print(file_names, prediction, file=out_f)

    return





### main ###

# train
if args.train == True:
    print("starting training pipeline")
    if args.preprocessed == False:
        print("using tree sequences")
        prep_trees_and_train()
    else:
        print("using pre-processed tensors")
        prep_preprocessed_and_train()

# predict
if args.predict == True:
    print("starting prediction pipeline")
    if args.empirical == None:
        print("predicting on simulated data")
        if args.preprocessed == True:
            print("using pre-processed tensors")
            prep_preprocessed_and_pred()
        else:
            print("using tree sequences")
            prep_trees_and_pred()
    else:
        print("predicting on empirical data")
        prep_empirical_and_pred()
