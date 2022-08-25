# checking over the command line params
import os


def check_params(args):

    # avoid overwriting saved weights or other output files
    if args.train == True:
        if os.path.exists(args.out + "_model.hdf5"):
            print("saved model with specified output name already exists (i.e. --out)")
            exit()
        if os.path.exists(f"{args.out}_training_mean_sd.npy"):
            print("saved mean and sd with specified output name already exists (i.e. --out)")
            exit()
    if args.predict == True:
        if os.path.exists(f"{args.out}_predictions.txt"):
            print("saved predictions with specified output name already exists (i.e. --out)")
            exit()

    # arguments for training
    if args.train == True:
        if args.num_snps == None:
            print("specify num snps via --num_snps")
            exit()
        if args.max_n == None:
            print("specify max sample size via --max_n")
            exit()

    # arguments for prediction
    if args.predict == True:
        if args.training_params == None:
            print("specify params file via --training_params")
            exit()

    # arguments related to preprocessing
    if args.preprocessed == True:
        if args.num_reps > 1:
            print("can't bootstrap on preprocessed data, only tree sequences or VCF")
            exit()

    # check some other param combinations
    if args.train == False and args.predict == False:
        print("either --train or --predict")
        exit()
    if args.sampling_width != None:
        if args.sampling_width > 1 or args.sampling_width <= 0:
            print("sampling width as proportion, (0,1)")
            exit()
    if args.predict == True and args.empirical == None:
        if args.num_pred != None:
            if args.num_pred % args.batch_size != 0:
                print(
                    "\n\npred sets each need to be divisible by batch_size; otherwise some batches will have missing data\n\n"
                )
                exit()
    if args.empirical == None and args.preprocessed == False:
        if args.edge_width == None:
            print("need to specify edge_width (via --edge_width)")
            exit()
    if (
        args.edge_width == 'sigma'
        and args.sampling_width != None
        and args.map_width != None
    ):
        print(
            "\n\nIf W and S are fixed, you must also fix the edge_width; otherwise the CNN can see sigma directly in the fourth input\n\n"
        )
        exit()
    if args.predict == True and args.preprocessed == False  and args.empirical == None:
        if args.min_n == None:
            print("missing min n, via --min_n")
            exit()
    if args.preprocessed == False and args.empirical == False:
        if args.mutate == None:
            print("specify whether or not to mutate the tree sequences")
            exit()
        elif args.recapitate == None:
            print("specify whether or not to recapitate the tree sequences")
            exit()


