
import os, json

def check_params(args):

    if args.out == None:
        print("specify output")
        exit()
    if args.num_snps == None:
        print("specify num snps")
        exit()
    if args.max_n == None:
        print("specify n")
        exit()


    # avoid overwriting saved weights or predsets                                
    if os.path.exists(args.out+"_model.hdf5"):
        print("saved model with specified name already  exists")
        exit()
    if os.path.exists(args.out+"_predsets"):
        print("saved predsets with specified name already exist")
        exit()
    if os.path.exists(args.out+'_params.json'):
        print("saved params with specified name already exist")
        exit()

    # check some more params                                                     
    if args.train == False and args.predict == False:
        print("either train or predict")
        exit()
    # if args.recapitate != "True" and args.recapitate != "False":
    #     print("specify recapitate option")
    #     exit()
    # if args.mutate != "True" and args.mutate != "False":
    #     print("specify mutate option")
    #     exit()
    if args.phase != 1 and args.phase != 2:
        print("incorrect phasing specification")
        exit()
    if args.polarize != 1 and args.polarize !=2:
        print("polarize argument incorrectly specified")
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
        if (args.num_pred % args.batch_size != 0):
            print("\n\npred sets each need to be divisible by batch_size; otherwise some batches will have missing data\n\n")
            exit()

    # avoiding information leakage through W and S parameters                                                                                 
    if args.edge_width == None and args.sampling_width != None and args.map_width != None:
         print("\n\nIf W and S are fixed, you must also fix the edge_width; otherwise the CNN can see sigma directly in the fourth input\n\n")
         exit()

    # in training, we normalize the targets; need to back-transform using the mean and sd from training                                       
    count_training_inputs=0
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

    # # save params
    # with open(args.out+'_params.json', 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)
    # f.close()
