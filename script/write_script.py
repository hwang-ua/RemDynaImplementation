def cgw():
    good = True

    line_in_file = 1
    count = 1000
    file = open("tasks_" + str(count) + ".sh", 'w')

    agent = "REM_Dyna"
    for offline in [1]:
        for num_near in [8]:
            for num_planning in [1]:
                for num_run in range(5):
                    for mode in [17]:
                        if mode == 0:
                            limit_list = [-0.2]#[-0.1]#[-0.2]#
                        elif mode in [10, 17]:
                            limit_list = [-0.1]#[-25.0]#[-30.0]#
                        elif mode == 21:
                            limit_list = [0.0]
                        else:
                            print("UNKNOWN MODE FOR LIMIT")

                        for limit in limit_list:

                            for kscale in [1.0]:#[1.0]:#
                                sim_list = [0.0]
                                for similarity in sim_list:
                                    for add_prot in [1]:
                                        if mode in [0, 11, 12, 13, 18]:
                                            fix_cov_list = [0.025]#[0.0001, 0.00025, 0.0005]
                                        elif mode in [1, 2, 10, 14, 15, 16, 17, 19, 20]:
                                            fix_cov_list = [0.025]
                                        elif mode in [21]:
                                            fix_cov_list = [0.0]
                                        else:
                                            print("UNKNOWN MODE FOR FIX_COV")

                                        for fix_cov in fix_cov_list:
                                            # limit = -0.015 * (1.0 / fix_cov)

                                            if mode in [2, 10, 11, 16]:
                                                opt_mode_list = [1]
                                            elif mode in [0, 1, 12, 13, 14, 15, 17, 18, 19, 20, 21]:
                                                opt_mode_list = [4]
                                            else:
                                                print("UNKNOWN MODE FOR OPTIMIZER")

                                            for opt_mode in opt_mode_list:
                                                for alg in ["Q"]:
                                                    for lambda_ in [0.0]:

                                                        if opt_mode in [0]:
                                                            momentum_list = [0.9, 0.99]
                                                        elif opt_mode in [4, 1]:
                                                            momentum_list = [0.0]

                                                        for momentum in momentum_list:

                                                            if opt_mode in [4]:
                                                                rms_list = [0.0]
                                                            elif opt_mode in [0, 1]:
                                                                rms_list = [0.999]

                                                            for rms in rms_list:
                                                                if opt_mode in [4]:
                                                                    alpha_list = ["0.001", "0.0003", "0.0001", "0.00003", "0.00001"]#[0.03125, 0.0625, 0.125, 0.25, 0.3]# [0.00097656, 0.00195312, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
                                                                elif opt_mode in [0, 1]:
                                                                    alpha_list = [0.01, 0.001, 0.0005]

                                                                for alpha in alpha_list:
                                                                    # for buffer_size in [50000]:
                                                                    for sync in [1]:
                                                                        file.write("python3 -m cProfile -o tasks_" + str(
                                                                            count) + "time.log " +
                                                                                   "experiment_cc.py" + " " +
                                                                                   agent + " " +
                                                                                   str(alpha) + " " +
                                                                                   str(num_near) + " " +
                                                                                   str(limit) + " " +
                                                                                   str(num_run) + " " +
                                                                                   str(mode) + " " +
                                                                                   str(kscale) + " " +
                                                                                   str(similarity) + " " +
                                                                                   str(add_prot) + " " +
                                                                                   str(fix_cov) + " " +
                                                                                   str(alg) + " " +
                                                                                   str(lambda_) + " " +
                                                                                   str(momentum) + " " +
                                                                                   str(rms) + " " +
                                                                                   str(opt_mode) + " " +
                                                                                   str(offline) + " " +
                                                                                   str(num_planning) + " " +
                                                                                   str("0.0") + " " +
                                                                                   str("1000") + " " +
                                                                                   str(sync) + "\n")
                                                                        count += 1
                                                                        if count % line_in_file == 0:
                                                                            file.close()
                                                                            file = open(
                                                                                "tasks_" + str(count // line_in_file) + ".sh", 'w')
                                                                            print("tasks_" + str(
                                                                                count // line_in_file) + ".sh" + " done")

def Q_learning():

    line_in_file = 1
    count = 2000
    file = open("tasks_" + str(count // line_in_file) + ".sh", 'w')

    offline = 0
    agent_list = ["Q_learning"]
    for agent in agent_list:
        for num_near in [0]:
            for num_planning in [0]:
                for limit in [0]:
                    for num_run in range(5):
                        for mode in [0]:
                            for kscale in [1e-7]:
                                for similarity in [0.0]:
                                    for add_prot in [1]:
                                        for fix_cov in [0]:

                                            if mode in [2]:
                                                opt_mode_list = [1]
                                            elif mode in [0, 3, 4]:
                                                opt_mode_list = [4]
                                            else:
                                                print("UNKNOWN MODE")

                                            for opt_mode in opt_mode_list:
                                                for alg in ["Q"]:
                                                    for lambda_ in [0.0]:
                                                        for momentum in [0.0]:

                                                            if opt_mode in [4]:
                                                                rms_list = [0.0]
                                                            elif opt_mode in [1]:
                                                                rms_list = [0.9, 0.99, 0.999]
                                                            else:
                                                                print("UNKNOWN MODE")

                                                            for rms in rms_list:
                                                                if opt_mode in [4]:
                                                                    alpha_list = [0.1, 0.2, 0.4, 0.8, 1.6]
                                                                elif opt_mode in [0, 1]:
                                                                    alpha_list = [0.1, 0.01, 0.001]

                                                                for alpha in alpha_list:
                                                                    file.write("python3 -m cProfile -o tasks_" + str(
                                                                        count) + "time.log " +
                                                                               "experiment_cc.py" + " " +
                                                                               agent + " " +
                                                                               str(alpha) + " " +
                                                                               str(num_near) + " " +
                                                                               str(limit) + " " +
                                                                               str(num_run) + " " +
                                                                               str(mode) + " " +
                                                                               str(kscale) + " " +
                                                                               str(similarity) + " " +
                                                                               str(add_prot) + " " +
                                                                               str(fix_cov) + " " +
                                                                               str(alg) + " " +
                                                                               str(lambda_) + " " +
                                                                               str(momentum) + " " +
                                                                               str(rms) + " " +
                                                                               str(opt_mode) + " "+
                                                                               str(offline) + " " +
                                                                               str(num_planning) + "\n")
                                                                    count += 1
                                                                    if count % line_in_file == 0:
                                                                        file.close()
                                                                        file = open(
                                                                            "tasks_" + str(count // line_in_file) + ".sh", 'w')
                                                                        print("tasks_" + str(
                                                                            count // line_in_file) + ".sh" + " done")


def lplc_rl():
    numf_list = [16, 32]
    lr_list = [0.00005, 0.0001, 0.0005, 0.001]
    batchsize_list = [128, 256]

    line_in_file = 1
    num_run = 1
    count = 0
    file = open("tasks_"+str(count)+".sh", 'w')

    for f in numf_list:
        for lr in lr_list:
            for size in batchsize_list:
                file.write("python3 -u feature_construction_rlpaper.py "+str(f)+" "+str(lr)+" "+str(size)+"\n")
                count += 1
                if count % line_in_file == 0:
                    file.close()
                    file = open("tasks_"+str(count//line_in_file)+".sh", 'w')
                    print("tasks_"+str(count//line_in_file)+".sh" + " done")

    file.close()

def auto_ecd():
    """
        gamma = map(float, sys.argv[2].strip('[]').split(','))
        gamma = gamma[0] if len(gamma) == 1 else gamma
        constraint = int(sys.argv[3])
        beta = float(sys.argv[4])
        delta = float(sys.argv[5])
        lr = float(sys.argv[6])
        lr_rcvs = lr
        num_epochs = int(sys.argv[7])
        num_epochs_rcvs = num_epochs
        scale = int(sys.argv[8])
        """
    continuous_list = [1]
    gamma_list = ["[0.998]", "[0.998,0.8]", "[0.998,0.99]"]
    constraint_list = [0, 1]
    beta_list = [1e-3, 1e-5, 1e-6]
    delta_list = [1]
    lr_list = [0.0001]
    num_epochs = [200]
    scale_list = [0, 1]

    line_in_file = 1
    count = 0
    file = open("tasks_" + str(count) + ".sh", 'w')

    for continuous in continuous_list:
        for gamma in gamma_list:
            for constraint in constraint_list:
                for beta in beta_list:
                    for delta in delta_list:
                        for lr in lr_list:
                            for epoch in num_epochs:
                                for scale in scale_list:
                                    file.write("python3 -u feature_construction.py "
                                               + str(continuous) + " "
                                               + str(gamma) + " "
                                               + str(constraint) + " "
                                               + str(beta) + " "
                                               + str(delta) + " "
                                               + str(lr) + " "
                                               + str(epoch) + " "
                                               + str(scale) + "\n")
                                    count += 1
                                    if count % line_in_file == 0:
                                        file.close()
                                        file = open("tasks_" + str(count // line_in_file) + ".sh", 'w')
                                        print("tasks_" + str(count // line_in_file) + ".sh" + " done")
    file.close()


def random_walk_cgw():
    num_ep = 50
    num_point = 200
    file_order = 25

    for o in range(file_order):
        file = open("tasks_" + str(o) + ".sh", 'w')
        file.write("python3 -u random_walk_data_cgw.py " + str(num_ep) + " " + str(num_point) + " " + str(o))
        file.close()
    return

def feature_construction():
    line_in_file = 1
    count = 1000
    file = open("tasks_" + str(count) + ".sh", 'w')

    for suc_prob in [1.0]:
        for input_range_low in [0.0]:
            for num_epoch in [1000]:
                for beta in [0.01, 0.1, 1.0]:
                    for feature in [32]:
                        for legalv in [0]:
                            for delta in [0.01, 0.1, 1.0]:
                                file.write("python feature_construction.py" + " " +
                                           str(suc_prob) + " " +
                                           str(input_range_low) + " " +
                                           str(num_epoch) + " " +
                                           str(beta) + " " +
                                           str(feature) + " " +
                                           str(legalv) + " " +
                                           str(delta) + "\n")
                                count += 1
                                if count % line_in_file == 0:
                                    file.close()
                                    file = open(
                                        "tasks_" + str(count // line_in_file) + ".sh", 'w')
                                    print("tasks_" + str(
                                        count // line_in_file) + ".sh" + " done")

def feature_construction_graph():
    line_in_file = 1
    count = 1000
    file = open("tasks_" + str(count) + ".sh", 'w')

    for beta in [0.1, 1e-3, 1e-5]:
        for delta in [0.1, 0.5, 1.0]:
            file.write("python3 feature_construction_graph.py" + " " +
                       str(beta) + " " +
                       str(delta) + "\n")
            count += 1
            if count % line_in_file == 0:
                file.close()
                file = open(
                    "tasks_" + str(count // line_in_file) + ".sh", 'w')
                print("tasks_" + str(
                    count // line_in_file) + ".sh" + " done")

def offline_NN():
    line_in_file = 1
    count = 1000
    file = open("tasks_" + str(count) + ".sh", 'w')

    for node in ["512,512,512"]:
        for lr in [0.0001]:
            for train_ep in [500]:
                if train_ep == 500:
                    epoch_list = [200]
                elif train_ep == 100:
                    epoch_list = [100, 500]
                for epoch in epoch_list:
                    for batch_size in [1024]:
                        for which in ["forward", "backward", "reward"]:
                            file.write("python3 construct_offline_NN_model.py" + " " +
                                       node + " " +
                                       str(lr) + " " +
                                       str(epoch) + " " +
                                       str(train_ep) + " " +
                                       str(which) + " " +
                                       str(batch_size) + "\n"
                                       )
                            count += 1
                            if count % line_in_file == 0:
                                file.close()
                                file = open("tasks_" + str(count // line_in_file) + ".sh", 'w')
                                print("tasks_" + str(count // line_in_file) + ".sh" + " done")


# random_walk_cgw()
# auto_ecd()
cgw()
# Q_learning()
# feature_construction()
# feature_construction_graph()
# offline_NN()
