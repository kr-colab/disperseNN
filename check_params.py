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

    # required arguments for training
    if args.train == True:
        pass

    # required arguments for prediction
    if args.predict == True:
        if args.training_mean_sd == None:
            print("specify training mean and sd via --training_mean_sd")
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
        if args.num_pred == None:
            print("how many pred sets? (via --num_pred)")
            exit()
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
                

