# checking over the command line params
import os


def check_params(args):

    # avoid overwriting saved weights
    if os.path.exists(args.out + "_model.hdf5"):
        print("saved model with specified name already  exists")
        exit()

    # check some other param combinations
    if args.train == False and args.predict == False:
        print("either train or predict")
        exit()
    if args.sampling_width != None:
        if args.sampling_width > 1 or args.sampling_width <= 0:
            print("sampling width as proportion, (0,1)")
            exit()
    if args.map_width != None and args.width_list != None:
        print("width and width lists are both provided; choose one or the other")
        exit()
    if args.predict == True:
        if args.num_pred == None:
            print("how many pred sets?")
            exit()
        if args.num_pred % args.batch_size != 0:
            print(
                "\n\npred sets each need to be divisible by batch_size; otherwise some batches will have missing data\n\n"
            )
            exit()
    if args.empirical == None and args.preprocessed == False:
        if args.edge_width == None:
            print("need to specify edge_width")
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
                
    # check that mean and sd are provided in exactly one form
    count_training_inputs = 0
    if args.train == True:
        count_training_inputs += 1
    if args.training_mean != None or args.training_sd != None:
        count_training_inputs += 1
    if args.training_targets != None:
        count_training_inputs += 1
    if count_training_inputs == 0:
        print("need to supply either training maps, or the mean and sd from training")
        exit()
    elif count_training_inputs > 1:
        print("multiple training inputs, unclear which to use")
        exit()
