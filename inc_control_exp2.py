#on-policy supervision, micture constraints

import numpy as np
from utils.parser import *
from utils.utils1 import *
from utils.utils2 import *
from models.nn_model import *
from models.utils_model import *
import torch
import torch.optim as O
import copy
import sys
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

def selectActionInc(dataset, numActions, state):
    if dataset in ['MC', 'SMC']:
        if np.random.uniform(1) < 0.1:
            action = np.random.randint(num_action)
        else:
            action = 2 if state[1]>=0 else 0
    elif dataset in ['PW', 'random_PW']:
        if np.random.uniform(1) < 0.5:
            action = 2
        else:
            action = 0
    elif dataset == "AC":
        action = np.random.randint(num_action)
    elif dataset == "CA":
        action = np.random.randint(num_action)
    elif dataset == "CA3":
        action = np.random.randint(num_action)
    return action


dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# sys.argv=['']
args = parse_args()
args = add_args(args)

old_loss = args.old_loss
supervision_loss = args.supervision_loss

#dataset parameters
dataset = args.dataset
normalize = args.normalize
scale_reward = args.scale

#run parameters
num_run = args.num_run
path = args.path
file_prefix = args.file_prefix

#representation parameters
reg_kl = args.reg_kl
reg_rec = args.reg_rec
reg_correlation = args.reg_correlation
reg_orthogonal = args.reg_orthogonal
reg_cosine = args.reg_cosine
reg_old_rep = args.reg_old_rep
target_old = args.target_old
copy_rate = args.copy_rate
theta = args.theta
lr_nn = args.lr_nn
lr_nn_decay_rate = args.lr_nn_decay_rate
num_epoch = args.num_epoch
save_rep = args.save_rep
eval_rep = args.eval_rep

#control parameters
train_control_parallel = args.train_control_parallel
train_control_after = args.train_control_after
lr_sarsa = args.lr_sarsa
epsilon = args.epsilon
max_steps = args.max_steps
max_steps_episode = args.max_steps_episode
lr_sarsa_decay_rate = args.lr_sarsa_decay_rate
epsilon_decay_rate = args.epsilon_decay_rate
lambdaa_sarsa = args.lambdaa_sarsa
retain_rep = args.retain_rep

#buffer parameters
use_data = args.use_data
lr_sarsa_data = args.lr_sarsa_data
epsilon_data = args.epsilon_data
epsilon_decay_rate_data = args.epsilon_decay_rate_data
batch_size = args.batch_size
do_buffer_refurbish = args.do_buffer_refurbish
buffer_refurbish = args.buffer_refurbish

buffer_size = args.buffer_size
random_replace = args.random_replace
use_reservoir = args.use_reservoir
use_prototypes = args.use_prototypes
use_custom_prototypes = args.use_custom_prototypes
custom_buffer_type_random = args.custom_buffer_type_random
use_raw_prototypes = args.use_raw_prototypes
prototype_lambda = args.prototype_lambda
prototype_epsilon = args.prototype_epsilon
use_on_policy = args.use_on_policy
use_Euclidean = args.use_Euclidean

#### Initialze a Env ####
if dataset == 'MC':
    from env.mc_env import mountaincar
    env = mountaincar(normalized=normalize, max_step=max_steps_episode+1) # max_steps + 1
    num_action = 3
    ae_shape = [2, 32, 256]
    observations = np.array([[-1.0,0.01],[-0.5,-0.05],[-0.5,0.05],[0.3,0.02]])
    observations = mc_normalization(observations)
    actions = [2,0,2,2]
    gamma = 1.0
    episodeMode = True

elif dataset == 'SMC':
    from env.mc_env import mountaincar
    env = mountaincar(normalized=normalize, max_step=max_steps_episode+1, sparse_reward=True) # max_steps + 1
    num_action = 3
    ae_shape = [2, 32, 256]
    observations = np.array([[-1.0,0.01],[-0.5,-0.05],[-0.5,0.05],[0.3,0.02]])
    observations = mc_normalization(observations)
    actions = [2,0,2,2]
    gamma = 0.998
    episodeMode = True

elif dataset in ['PW', 'random_PW']:
    if dataset == 'PW':
        from env.pw_env import puddleworld
        env = puddleworld(normalized=normalize)
    else:
        from env.random_env import random_puddleworld
        env = random_puddleworld(normalized=normalize)

    num_action = 4
    ae_shape = [2, 32, 256]
    observations = np.array([[0.05,0.2],[0.5,0.5],[0.05,0.95],[0.95,0.2],[0.95,0.85]])
    observations = pw_normalization(observations)
    actions = [2,0,1,2,0]
    gamma = 1.0
    episodeMode = True

elif dataset == 'AC':
    from env.ac_enc import acrobot
    env = acrobot(normalized=normalize)
    num_action = 3
    ae_shape = [4, 32, 256]
    observations = np.array([[0.34,7.0,0.93,-20.0],[0.34,7.0,0.93,20.0]])
    observations = ac_normalization(observations)
    actions = [0,2]
    gamma = 1.0
    episodeMode = True

elif dataset == 'CA':
    from env.ca_env import catcher
    env = catcher(init_lives=1)
    num_action = 2
    ae_shape = [4, 32, 256]
    observations = np.array([[50.0,5.0,10.0,5.0],[20.0,2.0,22.0,50.0]])
    observations = ca_normalization(observations)
    actions = [1,0]
    gamma = 1.0
    episodeMode = True

elif dataset == 'CA3':
    from env.ca_env import catcher3
    env = catcher3(init_lives=1)
    num_action = 3
    ae_shape = [4, 32, 256]
    gamma = 1.0
    episodeMode = True

else:
    assert False, 'unknown dataset'

observations = torch.from_numpy(observations).float()

#### Reading data ####
train_data = []
for i in range(1, num_run+1):
    data = create_train_dataset('data/{}/10ktrain{}.txt'.format(dataset, i+50), dataset, normalize=normalize, scale=scale_reward)
    train_data.append(data)

if supervision_loss in ['MSTDE', 'MSTDE_SR', 'EMB']:
    test_data = create_test_dataset('data/{}/test.txt'.format(dataset), dataset, normalize=normalize, scale=scale_reward)
elif supervision_loss == 'SR':
    test_data = create_train_dataset('data/{}/10ktest-trajectory.txt'.format(dataset), dataset, normalize=normalize, scale=scale_reward)

test_size = test_data[0].shape[0]

#Initialize bookkeeping variables
if train_control_parallel:
    train_control_test_num_episodes_parallel = np.zeros([num_run, num_epoch])
    train_control_test_avg_reward_per_episode_parallel = np.zeros([num_run, num_epoch])
if train_control_after:
    if episodeMode:
        train_control_train_num_steps_per_episode = np.zeros([num_run, max_steps])
        train_control_train_num_steps_per_episode.fill(np.nan)
        train_control_train_cumulative_reward_per_episode = np.zeros([num_run, max_steps])
        train_control_train_cumulative_reward_per_episode.fill(np.nan)
    else:
        train_control_train_cumulative_reward_per_step = np.zeros([num_run, max_steps])
    train_control_test_num_episodes = np.zeros([num_run])
    train_control_test_avg_reward_per_episode = np.zeros([num_run])


if episodeMode:
    num_steps_per_episode = np.zeros([num_run, num_epoch])
    num_steps_per_episode.fill(np.nan)
    cumulative_reward_per_episode = np.zeros([num_run, num_epoch])
    cumulative_reward_per_episode.fill(np.nan)
else:
    cumulative_reward_per_step = np.zeros([num_run, num_epoch])

#### Run model ####
for i in range(num_run):

    time = 0

    #Initialize a model
    model_rep = rep_model(shape=ae_shape)
    if supervision_loss == 'SR':
        model_rep_state = rep_state_model(shape=ae_shape)
        model_rep_reward = rep_reward_model(shape=ae_shape)
    else:
        model_rep_value = rep_value_model(shape=ae_shape)
    model_control = control_model(shape=ae_shape, num_action=num_action)
    if not use_data:
        model_control_incremental = control_model(shape=ae_shape, num_action=num_action)
        # optimizer_control_incremental = torch.optim.Adam(model_control_incremental.parameters(), lr=lr_sarsa_data, amsgrad=True, betas=(0.0, 0.9))
        optimizer_control_incremental = torch.optim.RMSprop(model_control_incremental.parameters(), lr=lr_sarsa_data)
        # optimizer_control_incremental = torch.optim.SGD(model_control_incremental.parameters(), lr=lr_sarsa_data)

        # model_rep_random = rep_random_model(shape=ae_shape)

    # optimizer_rep = torch.optim.Adam(model_rep.parameters(), lr=lr_nn, amsgrad=True, betas=(0.0, 0.9))
    # optimizer_rep = torch.optim.RMSprop(model_rep.parameters(), lr=lr_nn, alpha=0.9)
    optimizer_rep = torch.optim.SGD(model_rep.parameters(), lr=lr_nn)
    if supervision_loss == 'SR':
        optimizer_rep_state = torch.optim.RMSprop(model_rep_state.parameters(), lr=lr_nn, alpha=0.9)
        optimizer_rep_reward = torch.optim.RMSprop(model_rep_reward.parameters(), lr=lr_nn, alpha=0.9)
    else:
        # optimizer_rep_value = torch.optim.Adam(model_rep_value.parameters(), lr=lr_nn, amsgrad=True, betas=(0.0, 0.9))
        # optimizer_rep_value = torch.optim.RMSprop(model_rep_value.parameters(), lr=lr_nn, alpha=0.9)
        optimizer_rep_value = torch.optim.SGD(model_rep_value.parameters(), lr=lr_nn)


    # optimizer_rep = torch.optim.Adam(model_rep.parameters(), lr=lr_nn)
    # optimizer_rep_value = torch.optim.Adam(model_rep_value.parameters(), lr=lr_nn)

    optimizer_control = torch.optim.SGD(model_control.parameters(), lr=lr_sarsa)

    criterion = torch.nn.MSELoss(reduction='avg')

    env_local = copy.deepcopy(env)

    env_local_incremental = copy.deepcopy(env)
    # env_local.setseed(i)

    logging.info('_________________')
    logging.info('run {}'.format(i))

    data = train_data[i]
    train_size = buffer_size
    num_iter_per_epo = int(train_size/batch_size)
    if not use_data:
        num_iter_per_epo = args.num_iter_per_epo

    scheduler_rep = O.lr_scheduler.StepLR(optimizer_rep, step_size=1, gamma=lr_nn_decay_rate)
    if supervision_loss == 'SR':
        scheduler_rep_state = O.lr_scheduler.StepLR(optimizer_rep_state, step_size=1, gamma=lr_nn_decay_rate)
        scheduler_rep_reward = O.lr_scheduler.StepLR(optimizer_rep_reward, step_size=1, gamma=lr_nn_decay_rate)
    else:
        scheduler_rep_value = O.lr_scheduler.StepLR(optimizer_rep_value, step_size=1, gamma=lr_nn_decay_rate)

    #start a buffer
    if not use_data:
        buffer = replay_buffer(buffer_size, random_replace=random_replace, use_reservoir=use_reservoir, model_rep=model_rep, model_control=model_control_incremental, use_prototypes=use_prototypes, use_custom_prototypes=use_custom_prototypes, custom_buffer_type_random=custom_buffer_type_random, use_raw_prototypes=use_raw_prototypes, prototype_lambda=prototype_lambda, prototype_epsilon=prototype_epsilon, use_on_policy=use_on_policy, use_Euclidean=use_Euclidean)
        model_rep = model_rep.train(False)
        aborted = False
        total_episodes = 0
        current_steps = 0
        current_reward = 0
        epsilon_incremental = epsilon_data
        #start the interaction
        observation = env_local_incremental.reset()
        state = torch.from_numpy(observation).float()
        rep_state = model_rep(state).detach()
        # values = model_control_incremental(model_rep(state)).detach().data.numpy()
        values = model_control_incremental(rep_state).detach().data.numpy()
        action = pick_action(values, model_control_incremental.num_action, epsilon_incremental)
        # print('First {},{}'.format(values,action))
        model_rep = model_rep.train(True)
        traces = torch.zeros([1, ae_shape[-1]])
        traces = torch.zeros([1, model_control_incremental.num_action*ae_shape[-1]])
    else:
        buffer = replay_buffer(buffer_size, random_replace=random_replace, use_reservoir=use_reservoir)

    for ep in range(0,num_epoch):
        # print("epoch:{}".format(ep))
        insert_inBuffer = True
        if use_data:
            try:
                observation, observation_next, r_next, g_next = data[0][ep],data[1][ep],data[2][ep],data[3][ep]
            except:
                insert_inBuffer = False
        else:
            model_rep = model_rep.train(False)
            #take action, observe next state
            # current_val = model_control_incremental(model_rep(state))[action]
            if retain_rep:
                current_val = model_control_incremental(rep_state).detach().data.numpy()[action]
            else:
                current_val = model_control_incremental(model_rep(state)).detach().data.numpy()[action]
            observation_new, r_new, done, info = env_local_incremental.step(action)
            r_next = process_reward(dataset,scale_reward,r_new)
            current_reward += r_new
            current_steps += 1
            state_new = torch.from_numpy(observation_new).float()
            rep_state_new = model_rep(state_new).detach()
            # values_new = model_control_incremental(model_rep(state_new)).detach().data.numpy()
            values_new = model_control_incremental(rep_state_new).detach().data.numpy()
            action_new = pick_action(values_new, model_control_incremental.num_action, epsilon_incremental)
            # print('{},{}'.format(values_new,action_new))
            g_next = gamma

            if not episodeMode:
                cumulative_reward_per_step[i][ep] = total_reward

            if done or current_steps==max_steps_episode:
                if done:
                    target = r_new
                    g_next = 0.0
                else:
                    aborted = True
                # scheduler.step()
                epsilon_incremental *= epsilon_decay_rate_data
                logging.info('episode:{:.3f}, steps:{:.3f}, reward:{:.3f}, epoch:{}'.format(total_episodes, current_steps, current_reward, ep))
                #print bootstrap values
                bootstrapValues(model_rep, None, model_control_incremental, observations, actions)
                if episodeMode:
                    cumulative_reward_per_episode[i][total_episodes] = current_reward
                    num_steps_per_episode[i][total_episodes] = current_steps
                total_episodes += 1
                current_reward = 0
                current_steps = 0
            if not done:
                next_val = values_new[action_new]
                target = r_new+(gamma*next_val)

            temp_traces = traces[:,(ae_shape[-1]*action):(ae_shape[-1]*(action+1))].squeeze()
            # temp_traces = temp_traces + model_rep(state).detach()
            if retain_rep:
                temp_traces = temp_traces + rep_state
            else:
                temp_traces = temp_traces + model_rep(state).detach()
            traces[:,(ae_shape[-1]*action):(ae_shape[-1]*(action+1))].squeeze().copy_(temp_traces)

            optimizer_control_incremental.zero_grad()
            # print(current_val-target, current_val, target, torch.norm(model_rep(state).detach()), torch.norm(model_rep(state_new).detach()), r_new)
            # print(state, state_new, action_new)
            # loss = criterion(target,current_val)
            # loss.backward()
            for act in range(model_control_incremental.num_action):
                loss = 0.0*torch.norm(model_control_incremental.weights_sarsa._modules[str(act)].weight)
                loss.backward(torch.FloatTensor(np.asarray([0.0])),retain_graph=True)
                temp_traces = traces[:,(ae_shape[-1]*act):(ae_shape[-1]*(act+1))].squeeze()
                model_control_incremental.weights_sarsa._modules[str(act)].weight.grad += torch.mul(temp_traces,current_val-target)
            optimizer_control_incremental.step()
            model_control_incremental.check_parameters()

            if done or aborted:
                traces = torch.mul(traces,0.0)
            else:
                traces = torch.mul(traces,gamma*lambdaa_sarsa)

        if not use_data:
            observation_next = observation_new
            if aborted and not episodeMode:
                observation_new = env_local_incremental.reset()
                state_new = torch.from_numpy(observation_new).float()
                rep_state_new = model_rep(state_new).detach()
                # values_new = model_control_incremental(model_rep(state_new)).detach().data.numpy()
                values_new = model_control_incremental(rep_state_new).detach().data.numpy()
                action_new = pick_action(values_new, model_control_incremental.num_action, epsilon_incremental)
                # print('First {},{}'.format(values_new,action_new))

                aborted = False

            state_backup = state
            state = state_new
            rep_state = rep_state_new
            action = action_new
            model_rep = model_rep.train(True)

        if not buffer.is_full:
        # if ep == 0:
            if insert_inBuffer:
                # if use_data or not use_raw_prototypes:
                if use_data:
                    buffer.add([observation,observation_next,r_next,g_next])
                else:
                    if not use_prototypes or use_custom_prototypes:
                        buffer.add([observation,observation_next,r_next,g_next])
                    else:
                        if use_raw_prototypes:
                            sample_rep = state_backup.detach().numpy()
                        else:
                            sample_rep = model_rep(state_backup).detach().numpy()
                            # sample_rep = model_rep_random(state_backup).detach().numpy()
                        buffer.add([observation,observation_next,r_next,g_next,sample_rep])
                    observation = observation_new
            continue

        #train representation with minibatch
        # logging.info('_________________')
        # logging.info('epoch {}'.format(ep))

        # scheduler_rep.step()
        # if supervision_loss == 'SR':
        #     scheduler_rep_state.step()
        #     scheduler_rep_reward.step()
        # else:
        #     scheduler_rep_value.step()

        # buffer.check_rep(model_rep)

        for j in range(0, num_iter_per_epo):

            #on-policy for supervision
            samples = buffer.sample(batch_size)

            current_state = torch.from_numpy(samples[0]).float()
            next_state = torch.from_numpy(samples[1]).float()
            r = torch.from_numpy(samples[2]).float()
            r = r[:,np.newaxis]
            g = torch.from_numpy(samples[3]).float()
            g = g[:,np.newaxis]

            optimizer_rep.zero_grad()
            if supervision_loss == 'SR':
                optimizer_rep_state.zero_grad()
                optimizer_rep_reward.zero_grad()
            else:
                optimizer_rep_value.zero_grad()

            current_rep = model_rep(current_state)
            act_current_rep = model_rep.activation.copy()
            if supervision_loss == 'SR':
                next_state_pred = model_rep_state(current_rep)
                next_reward_pred = model_rep_reward(current_rep)

                # loss = sr_loss(next_state_pred, next_reward_pred, r, criterion)
                loss_s = ((next_state_pred-next_state)**2).sum(dim=1).mean()
                loss_r = ((next_reward_pred-r)**2).mean()

                loss_s.backward(retain_graph=True)
                loss_r.backward(retain_graph=True)
            else:
                current_value = model_rep_value(current_rep)

                if not target_old:
                    next_rep = model_rep(next_state)
                    next_value = model_rep_value(next_rep)
                else:
                    next_rep = model_rep.forward_old(next_state)
                    next_value = model_rep_value.forward_old(next_rep)

                loss = mstde_loss(current_value, next_value, r, g, criterion)

                # loss.backward()
                loss.backward(retain_graph=True)

            #alternating optimization
            # optimizer_rep.step()
            # if supervision_loss == 'SR':
            #     optimizer_rep_state.step()
            #     optimizer_rep_reward.step()
            # elif supervision_loss == 'MSTDE':
            #     optimizer_rep_value.step()
            #
            #
            # optimizer_rep.zero_grad()
            # if supervision_loss == 'SR':
            #     optimizer_rep_state.zero_grad()
            #     optimizer_rep_reward.zero_grad()
            # else:
            #     optimizer_rep_value.zero_grad()

            #mixture for constraints
            if use_data or not use_prototypes:
                # samples = buffer.sample(batch_size)
                samples2 = buffer.sample(batch_size)
            else:
                if use_raw_prototypes:
                    sample_rep = state_backup.detach().numpy()
                else:
                    sample_rep = model_rep(state_backup).detach().numpy()
                    # sample_rep = model_rep_random(state_backup).detach().numpy()
                # samples = buffer.sample(batch_size,current_rep=sample_rep)
                samples2 = buffer.sample(batch_size,current_rep=sample_rep)

            # current_state = torch.from_numpy(samples[0]).float()
            # current_rep = model_rep(current_state)
            # act_current_rep = model_rep.activation.copy()
            other_state = torch.from_numpy(samples2[0]).float()

            current_rep_old = model_rep.forward_old(current_state)
            other_rep_old = model_rep.forward_old(other_state)

            #sparsity loss
            if reg_kl != 0.0:
                sparsity_loss(current_rep, theta, reg_kl)
                # loss_theta = sparsity_loss_all(model_rep.num_layers, act_current_rep, theta, reg_kl)

            if not old_loss:
                #build graph components
                other_rep = model_rep(other_state)
                act_other_rep = model_rep.activation.copy()
                # other_value = model_rep_value(other_rep)

                #cosine loss
                if reg_cosine != 0.0:
                    cosine_loss(current_rep, other_rep, reg_cosine)

                # uncorrelated loss
                if reg_correlation != 0.0:
                    uncorrelated_loss(current_rep, other_rep, reg_correlation)
                    # loss_cor = uncorrelated_loss(current_rep, other_rep, reg_correlation, buffer.kernel, current_state, other_state, use_raw_prototypes)
                    # # loss_cor = uncorrelated_loss_all(model_rep.num_layers, act_current_rep, act_other_rep, reg_correlation)

                #orthogonality loss
                if (reg_correlation*reg_orthogonal) != 0.0:
                    orthogonality_loss(current_rep, other_rep, reg_correlation, reg_orthogonal)
                    # loss_orthogonal = orthogonality_loss_all(model_rep.num_layers, act_current_rep, act_other_rep, reg_correlation, reg_orthogonal)

                if reg_old_rep != 0.0:
                    if use_on_policy:
                        old_rep_loss(current_rep_old, current_rep, reg_old_rep)
                    else:
                        old_rep_loss(other_rep_old, other_rep, reg_old_rep)

                # logging.info('loss:{},loss_theta:{},loss_cor:{},loss_orthogonal:{}'.format(loss.item(),loss_theta.item(),loss_cor.item(),loss_orthogonal.item()))

            # print("gradients")
            # model_rep.accumulate_gradient()

            loss=0.0*(current_rep.sum())
            loss.backward()

            optimizer_rep.step()
            if reg_old_rep != 0.0:
                time += 1
                if time % copy_rate == 0:
                    time = 0
                    model_rep.copy_network()
                    model_rep_value.copy_network()
            if supervision_loss == 'SR':
                optimizer_rep_state.step()
                optimizer_rep_reward.step()
            elif supervision_loss == 'MSTDE':
                optimizer_rep_value.step()

            model_rep.check_parameters()
            if supervision_loss == 'SR':
                model_rep_state.check_parameters()
                model_rep_reward.check_parameters()
            elif supervision_loss == 'MSTDE':
                model_rep_value.check_parameters()

        # if not use_data:
        if do_buffer_refurbish and not use_data and ep%buffer_refurbish == 0:
            model_rep = model_rep.train(False)
            #update buffer for all states
            for sampleNum in range(len(buffer.buffer)):
                buffer.buffer_rep_prototype[sampleNum] = model_rep(torch.from_numpy(buffer.buffer_prototype[sampleNum][0]).float()).detach().numpy()
            buffer.populate_kernel()
            model_rep = model_rep.train(False)

        if insert_inBuffer:
            # if use_data or not use_raw_prototypes:
            if use_data:
                buffer.add([observation,observation_next,r_next,g_next])
            else:
                if not use_prototypes or use_custom_prototypes:
                    buffer.add([observation,observation_next,r_next,g_next])
                else:
                    if use_raw_prototypes:
                        sample_rep = state_backup.detach().numpy()
                    else:
                        sample_rep = model_rep(state_backup).detach().numpy()
                        # sample_rep = model_rep_random(state_backup).detach().numpy()
                    buffer.add([observation,observation_next,r_next,g_next,sample_rep])
                observation = observation_new

        #evaluate learning
        # if eval_rep:
        if eval_rep or ep%1000 == 0:
            if supervision_loss == 'SR':
                #evaluate training MSTDE
                sre = sre_batch_data(criterion, model_rep_state, model_rep_reward, model_rep, data)
                logging.info('training SRE:{:.3f}'.format(sre))
                #evaluate testing RMSE
                sre = sre_batch_data(criterion, model_rep_state, model_rep_reward, model_rep, test_data)
                logging.info('testing SRE:{:.3f}'.format(sre))
            else:
                #evaluate training MSTDE
                mstde = mstde_batch_data(criterion, model_rep_value, model_rep, data)
                logging.info('training MSTDE:{:.3f}'.format(mstde))
                #evaluate testing RMSE
                rmse = rmse_test_data(criterion, model_rep_value, model_rep, test_data)
                logging.info('testing RMSE:{:.3f}'.format(rmse))
            #evaluate representation on test data
            model_rep = model_rep.eval()
            rep = model_rep(torch.from_numpy(test_data[0])).detach().numpy()
            sparsity_stat(rep, print_out=True)
            model_rep = model_rep.train(True)

        # rep_plot = rep[np.random.randint(rep.shape[0], size=100)]
        # plt.matshow(rep_plot)
        # minVal = rep_plot.min()
        # maxVal = rep_plot.max()
        # cmap = plt.cm.Blues
        # norm = plt.colors.Normalize(vmin=minVal, vmax=maxVal)
        # bar = plt.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,orientation='horizontal')
        # plt.show()

        if train_control_parallel:
            num_episodes, avg_reward_per_episode = train_control_do(criterion, optimizer_control, model_control, model_rep, env_local, gamma, max_steps, epsilon, max_steps_episode, lr_sarsa_decay_rate, epsilon_decay_rate)
            train_control_test_num_episodes_parallel[i][ep] = num_episodes
            train_control_test_avg_reward_per_episode_parallel[i][ep] = avg_reward_per_episode

    # if train_prediction_parallel:
    #     #print parallel prediction summary stats
    #     logging.info('mstde:mean={},std.error={}'.format(performance_stats(train_pred_mstde_parallel[i])))
    #     logging.info('rmse:mean={},std.error={}'.format(performance_stats(train_pred_test_rmse_parallel[i])))

    logging.info('_________________')
    logging.info('after representation learning'.format(ep))

    if train_control_after:
        if episodeMode:
            num_episodes, avg_reward_per_episode = train_control_do(criterion, optimizer_control, model_control, model_rep, env_local, gamma, max_steps, epsilon, max_steps_episode, lr_sarsa_decay_rate, epsilon_decay_rate, True, episodeMode, train_control_train_num_steps_per_episode[i], train_control_train_cumulative_reward_per_episode[i], None)
        else:
            num_episodes, avg_reward_per_episode = train_control_do(criterion, optimizer_control, model_control, model_rep, env_local, gamma, max_steps, epsilon, max_steps_episode, lr_sarsa_decay_rate, epsilon_decay_rate, True, episodeMode, None, None, train_control_train_cumulative_reward_per_step[i])
        train_control_test_num_episodes[i] = num_episodes
        train_control_test_avg_reward_per_episode[i] = avg_reward_per_episode


#print after control on-policy summary stats
logging.info('_________________')
logging.info('Control/representation performance:')
if episodeMode:
    mean_reward,stderr_reward,mean_steps,stderr_steps = performance_stats_on_policy(num_epoch, cumulative_reward_per_episode, num_steps_per_episode)
    saveArrayToFile(path+file_prefix+"_steps_joint", mean_steps)
    saveArrayToFile(path+file_prefix+"_steps_joint_stdErr", stderr_steps)
else:
    mean_reward,stderr_reward = performance_stats_on_policy(num_epoch, cumulative_reward_per_episode)
saveArrayToFile(path+file_prefix+"_reward_joint", mean_reward)
saveArrayToFile(path+file_prefix+"_reward_joint_stdErr", stderr_reward)

if train_control_after:
    #print after control on-policy summary stats
    logging.info('_________________')
    logging.info('On-policy performance:')
    if episodeMode:
        mean_reward,stderr_reward,mean_steps,stderr_steps = performance_stats_on_policy(max_steps, train_control_train_cumulative_reward_per_episode, train_control_train_num_steps_per_episode)
        saveArrayToFile(path+file_prefix+"_steps", mean_steps)
        saveArrayToFile(path+file_prefix+"_steps_stdErr", stderr_steps)
    else:
        mean_reward,stderr_reward = performance_stats_on_policy(max_steps, train_control_train_cumulative_reward_per_episode)
    saveArrayToFile(path+file_prefix+"_reward", mean_reward)
    saveArrayToFile(path+file_prefix+"_reward_stdErr", stderr_reward)
