import sys
from os.path import expanduser
import math
import numpy as np
import matplotlib.pyplot as plt
import os

color_idx = np.linspace(0, 1, 6)
def getColor(algoIndex):
    newColor = plt.cm.plasma_r(color_idx[algoIndex])
    return newColor

def exponential_smooth(data, beta):
    J = 0
    new_data = np.zeros(len(data))
    for idx in range(len(data)):
        J *= (1-beta)
        J += beta
        rate = beta / J
        if idx == 0:
            new_data[idx] = data[idx] * rate
        else:
            new_data[idx] = data[idx] * rate + new_data[idx - 1] * (1 - rate)
    return new_data

def draw(data):
    print(data.shape)
    # accumulate reward
    d = data
    learning = []
    for l in d:
        start = 0
        while start < len(l):
            if l[start] != 0:
                break
            else:
                start += 1
        print("this run starts from", start)
        # learning.append(l[start: start + 9200])
        learning.append(l[start: start + 100000])
    learning = np.array(learning)

    md = np.mean(learning, axis=0)
    ste = np.abs(np.std(learning, axis=0)) / np.sqrt(len(learning))
    upper = md + ste * 1
    lower = md - ste * 1
    return md, upper, lower

def pw_collect(pref):
    num_run = 100
    num_ep = 200
    return_rec = np.zeros((num_run, num_ep))
    not_exist = []
    for run in range(num_run):
        accum_r = pref + "_run" + str(run) + ".npy"
        step_per_ep = pref + "_run" + str(run) + "_stepPerEp.npy"

        if os.path.isfile(accum_r):
            reward_record = np.load(accum_r)
            step_record = np.load(step_per_ep)
            # print(step_record)
            total_s = -1
            last_ep = 0
            r_ep = []
            for s_idx in range(len(step_record)):
                s = step_record[s_idx]
                total_s += s
                if last_ep != 0:
                    r_ep.append(reward_record[total_s] - reward_record[last_ep])
                    # return_rec[run, s_idx] = reward_record[total_s] - reward_record[last_ep]
                else:
                    r_ep.append(reward_record[total_s])
                    # return_rec[run, s_idx] = reward_record[total_s]
                last_ep = total_s
            r_ep = exponential_smooth(r_ep, beta=0.1)
            length = len(r_ep)
            return_rec[run, :length] = r_ep
        else:
            not_exist.append(accum_r)

    cut = 0
    stop = False
    while cut < num_ep and not stop:
        k_ep = return_rec[:, cut]
        count = len(np.where(k_ep != 0)[0])
        if count >= num_run/2:
            cut += 1
        else:
            stop = True
    print("Not exist: \n", not_exist, "\n")
    return return_rec, cut

def bw():
    raw = "exp_result/paper/BW_ER/" + \
          "REM_Dyna_mode0_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
          "random_BufferOnly_alpha3e-05_divAF0_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_300000x100.npy"
    case = "ER"

    if case == "ER":
        new_rep = "exp_result/paper/BW_ER/" + \
                  "REM_Dyna_mode17_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
                  "random_BufferOnly_alpha3e-05_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_300000x100.npy"
        noCons = "exp_result/paper/noCons_BW_ER/" + \
                     "REM_Dyna_mode17noCons_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
                     "random_BufferOnly_alpha0.0003_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_500000x100.npy"
        ae = "exp_result/paper/AE_BW_ER/" + \
             "REM_Dyna_mode17AE_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
             "random_BufferOnly_alpha3e-05_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_500000x100.npy"

    elif case == "LLM":

        new_rep = "exp_result/paper/BW_LLM/" + \
                  "REM_Dyna_mode17_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
                  "random_alpha0.0003_divAF17_near8_protLimit-6.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_500000x100.npy"

        noCons = "exp_result/paper/noCons_BW_LLM/" + \
                 "REM_Dyna_mode17noCons_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
                 "random_alpha3e-05_divAF17_near8_protLimit-0.3_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_500000x100.npy"
        ae = "exp_result/paper/AE_BW_LLM/" + \
             "REM_Dyna_mode17AE_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
             "random_alpha8e-06_divAF17_near8_protLimit-0.1_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_500000x100.npy"

    elif case == "REM":
        new_rep = "exp_result/paper/BW_REM/" + \
                  "REM_Dyna_mode17_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
                  "random_alpha3e-05_divAF17_near8_protLimit-130.0_similarity0_sampleLimit0.0_kscale1e-07_fixCov0.0_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_300000x100.npy"

        noCons = "exp_result/paper/noCons_BW_REM/" \
                 "REM_Dyna_mode17noCons_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" + \
                 "random_alpha1e-05_divAF17_near8_protLimit-90.0_similarity0_sampleLimit0.0_kscale1e-07_fixCov0.0_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_500000x100.npy"
        ae = "exp_result/paper/AE_BW_REM/" + \
             "REM_Dyna_mode17AE_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
             "random_alpha1e-05_divAF17_near8_protLimit-70.0_similarity0_sampleLimit0.0_kscale1e-07_fixCov0.0_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_500000x100.npy"

    raw = np.load(raw)
    new_rep = np.load(new_rep)
    noCons = np.load(noCons)
    ae = np.load(ae)

    all_data = [raw, ae, noCons, new_rep]
    label = ["raw", "ae", "noCons", "new_rep"]
    # all_data = [raw, new_rep]
    cut = 10000
    fig = plt.figure()
    plt.xlim(0, cut)
    # plt.ylim(0, 2500)
    for i in range(len(all_data)):
        data = all_data[i]#[:, :cut]
        md, upper, lower = draw(data)
        x = np.linspace(0, len(md), len(md))
        c = getColor(i)
        plt.plot(x, md, color=c, label=label[i])
        plt.fill_between(x, upper, lower, facecolor=c, alpha=0.3)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    # fig.savefig("../BW_" + str(case) + ".pdf", dpi=300)


def pw():
    raw = "exp_result/paper/PW_ER/" + \
          "REM_Dyna_mode0_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
          "random_BufferOnly_alpha3e-05_divAF0_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"

    case = "ER"

    if case == "ER":
        new_rep = "exp_result/paper/PW_ER/" + \
                  "REM_Dyna_mode17_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
                  "random_BufferOnly_alpha3e-05_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"
        noCons = "exp_result/paper/noCons_PW_ER/" \
                 "REM_Dyna_mode17noCons_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" + \
                 "random_BufferOnly_alpha8e-06_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"
        ae = "exp_result/paper/AE_PW_ER/" + \
             "REM_Dyna_mode17AE_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
             "random_BufferOnly_alpha8e-06_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"
    elif case == "LLM":
        new_rep = "exp_result/paper/PW_LLM/" + \
                  "REM_Dyna_mode17_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
                  "random_alpha1e-05_divAF17_near8_protLimit-4.5_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"
        noCons = "exp_result/paper/noCons_PW_LLM/" \
                 "REM_Dyna_mode17noCons_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
                 "random_alpha8e-06_divAF17_near8_protLimit-0.6_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"
        ae = "exp_result/paper/AE_PW_LLM/" \
             "REM_Dyna_mode17AE_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" + \
             "random_alpha8e-06_divAF17_near8_protLimit-0.1_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"
    elif case == "REM":
        new_rep = "exp_result/paper/PW_REM/" \
                  "REM_Dyna_mode17_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
                  "random_alpha4e-06_divAF17_near8_protLimit-140.0_similarity0_sampleLimit0.0_kscale1e-07_fixCov0.0_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"
        noCons = "exp_result/paper/noCons_PW_REM/" \
                 "REM_Dyna_mode17noCons_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" + \
                 "random_alpha8e-06_divAF17_near8_protLimit-90.0_similarity0_sampleLimit0.0_kscale1e-07_fixCov0.0_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"
        ae = "exp_result/paper/AE_PW_REM/" \
             "REM_Dyna_mode17AE_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
             "random_alpha8e-06_divAF17_near8_protLimit-60.0_similarity0_sampleLimit0.0_kscale1e-07_fixCov0.0_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"

    all_data = [raw, new_rep, noCons, ae]
    label = ["raw", "new_rep", "noCons", "ae"]
    fig = plt.figure()
    plt.ylim(-500, 0)
    plt.xlim(0, 200)
    for i in range(len(all_data)):
        pref = all_data[i]
        all_run, cut = pw_collect(pref)
        print(i, cut)
        all_run = all_run * 40
        mean = np.zeros(cut)
        upper = np.zeros(cut)
        lower = np.zeros(cut)
        for ep in range(cut):
            kth_ep = all_run[:, ep]
            non0idx = np.where(kth_ep != 0)[0]
            learning = kth_ep[non0idx]
            mean[ep] = np.mean(learning)
            ste = np.abs(np.std(learning, axis=0)) / np.sqrt(len(learning))
            upper[ep] = mean[ep] + ste
            lower[ep] = mean[ep] - ste

        c = getColor(i)
        x = np.linspace(0, len(mean), len(mean))
        plt.plot(x, mean, color=c, label=label[i])
        plt.fill_between(x, upper, lower, facecolor=c, alpha=0.3)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=2, mode="expand", borderaxespad=0.)
    # plt.show()
    fig.savefig("../PW_" + str(case) + ".pdf", dpi=300)

def check_bw():
    raw = "exp_result/BW_ER/" + \
          "REM_Dyna_mode0_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
          "random_BufferOnly_alpha3e-05_divAF0_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_300000x100.npy"
    new_rep1 = "exp_result/illegalv_BW_ER/" \
              "REM_Dyna_mode17new_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
              "random_BufferOnly_alpha0.001_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_500000x5.npy"
    new_rep2 = "exp_result/illegalv_BW_ER/" \
              "REM_Dyna_mode17new_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
              "random_BufferOnly_alpha3e-05_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_500000x5.npy"
    new_rep3 = "exp_result/legalv_BW_ER/" + \
              "REM_Dyna_mode17new_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
              "random_BufferOnly_alpha0.001_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_500000x5.npy"
    noCons = "exp_result/noCons_BW_ER/" + \
                 "REM_Dyna_mode17noCons_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
                 "random_BufferOnly_alpha0.0003_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_500000x100.npy"
    ae = "exp_result/AE_BW_ER/" + \
         "REM_Dyna_mode17AE_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
         "random_BufferOnly_alpha3e-05_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_500000x100.npy"

    # all_data = [raw, ae, noCons, new_rep1, new_rep2, new_rep3]
    # label = ["raw", "ae", "supervised", "illegalv lr=0.0003", "illegalv lr=3e-05", "legalv lr=0.0003"]
    all_data = [raw, ae, noCons, new_rep1, new_rep3]
    label = ["raw", "ae", "supervised", "illegalv lr=0.001", "legalv lr=0.0003"]

    for i in range(len(all_data)):
        all_data[i] = np.load(all_data[i])

    # all_data = [raw, new_rep]
    cut = 10000#100000
    fig = plt.figure()
    plt.xlim(0, cut)
    # plt.ylim(0, 2500)
    for i in range(len(all_data)):
        data = np.array(all_data[i])
        print("\n", i)
        md, upper, lower = draw(data)
        x = np.linspace(0, len(md), len(md))
        c = getColor(i)
        plt.plot(x, md, color=c, label=label[i])
        plt.fill_between(x, upper, lower, facecolor=c, alpha=0.3)
        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #            ncol=2, mode="expand", borderaxespad=0.)
        plt.legend()
    plt.show()
    # fig.savefig("../BW_" + str(case) + ".pdf", dpi=300)


def check_pw():
    raw = "exp_result/paper/PW_ER/" + \
          "REM_Dyna_mode0_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
          "random_BufferOnly_alpha3e-05_divAF0_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"
    new_rep = "exp_result/paper/PW_ER/" + \
              "REM_Dyna_mode17_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
              "random_BufferOnly_alpha3e-05_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"
    nocons = "exp_result/paper/noCons_PW_ER/" \
             "REM_Dyna_mode17noCons_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
             "random_BufferOnly_alpha0.001_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"
    ae = "exp_result/paper/AE_PW_ER/" \
         "REM_Dyna_mode17AE_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
         "random_BufferOnly_alpha0.001_divAF17_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"

    # all_data = [raw, new_rep, nocons, ae]
    all_data = [new_rep, nocons, ae]
    label = ["raw", "new_rep", "noCons", "ae"]
    fig = plt.figure()
    # plt.ylim(-500, 0)
    plt.xlim(0, 200)
    for i in range(len(all_data)):
        pref = all_data[i]
        all_run, cut = pw_collect(pref)
        print(i, cut)
        all_run = all_run * 40
        mean = np.zeros(cut)
        upper = np.zeros(cut)
        lower = np.zeros(cut)
        for ep in range(cut):
            kth_ep = all_run[:, ep]
            non0idx = np.where(kth_ep != 0)[0]
            learning = kth_ep[non0idx]
            mean[ep] = np.mean(learning)
            ste = np.abs(np.std(learning, axis=0)) / np.sqrt(len(learning))
            upper[ep] = mean[ep] + ste
            lower[ep] = mean[ep] - ste

        c = getColor(i)
        x = np.linspace(0, len(mean), len(mean))
        plt.plot(x, mean, color=c, label=label[i])
        plt.fill_between(x, upper, lower, facecolor=c, alpha=0.3)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    # fig.savefig("../PW_" + str(case) + ".pdf", dpi=300)



# bw()
# pw()
check_bw()