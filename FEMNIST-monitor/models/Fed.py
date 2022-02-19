import copy
import torch
import numpy as np
from scipy.linalg import solve

import logging

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def outlier_detect(w_global, w_local, itera):
    # fc1 (26,512)
    w_global = w_global['resnet.fc1.weight'].cpu().numpy()
    # logging.info(f"w_global {w_global.shape}")
    w = []
    for i in range(len(w_local)):
        temp = (w_local[i]['resnet.fc1.weight'].cpu().numpy() - w_global) * 100
        w.append(temp)
    res = search_neuron_new(w)
    return res

def search_neuron_new(w):
    w = np.array(w)
    # w.shape (26, 26, 512)
    pos_res = np.zeros((len(w), 26, 512))
    for i in range(w.shape[1]):
        # all w of ratio
        for j in range(w.shape[2]):
            # one fc w
            temp = []
            for p in range(len(w)):
                # w[p, i, j] -0.11165216565132141
                temp.append(w[p, i, j])
                # logging.info(f"w[p, i, j] {w[p, i, j]}")
            # max_index 9
            max_index = temp.index(max(temp))
            # logging.info(f"max_index {max_index}")
            # pos_res[max_index, i, j] = 1

            '''
            np.abs(temp):
            [0.10568462 0.10679625 0.11067726 0.11750869 0.11332259 0.12109801
             0.11257678 0.10836944 0.10239743 1.9174933  0.10324977 0.1028914
             0.11362322 0.11100769 0.10127872 0.10815635 0.10976717 0.11223927
             0.11203289 0.11357106 0.10985509 0.11535287 0.10821186 0.12045987
             0.11170693 0.11165217]
             
            abs(w[max_index, i, j]):
            1.9174933433532715
            
            np.abs(temp) / abs(w[max_index, i, j]) > 0.8 :
            [False False False False False False False False False False False False
             False False False False False False False False False False False False
             False  True]
            
            '''


            outlier = np.where(np.abs(temp) / abs(w[max_index, i, j]) > 0.8)
            # logging.info(f"np.abs(temp) {np.abs(temp) / abs(w[max_index, i, j]) > 0.8}")
            # logging.info(f"abs(w[max_index, i, j]) {abs(w[max_index, i, j])}")
            # logging.info(f"outlier {outlier}")
            # logging.info(f"outlier[0] {outlier[0]}")

            # outlier[0] 25
            if len(outlier[0]) < 2:
                pos_res[max_index, i, j] = 1
    logging.info(f"pos_res {pos_res}")
    return pos_res

def whole_determination(pos, w_glob_last, cc_net):
    ratio_res = []
    for it in range(26):
        cc_class = it
        aux_sum = 0
        aux_other_sum = 0
        layer = 1
        for i in range(pos.shape[1]):
            for j in range(pos.shape[2]):
                if pos[cc_class, i, j] == 1:
                    temp = []
                    last = w_glob_last['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j]
                    cc = cc_net[cc_class]['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j]
                    for p in range(len(cc_net)):
                        temp.append(cc_net[p]['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j] - last)
                    temp = np.array(temp)
                    temp = np.delete(temp, cc_class)
                    temp_ave = np.sum(temp) / (len(cc_net) - 1)
                    aux_sum += cc - last
                    aux_other_sum += temp_ave
        if aux_other_sum != 0:
            res = abs(aux_sum) / abs(aux_other_sum)
        else:
            res = 10
        logging.info('label {}-----aux_data:{}, aux_other:{}, ratio:{}'.format(it, aux_sum, aux_other_sum, res))
        ratio_res.append(res)

    # normalize the radio alpha
    ratio_min = np.min(ratio_res)
    ratio_max = np.max(ratio_res)
    for i in range(len(ratio_res)):
        ratio_res[i] = 1.0 + 0.1 * ratio_res[i]
    return ratio_res

def monitoring(cc_net, pos, w_glob_last, w_glob, num_class, num_users, num_samples, args):
    res_monitor = []
    res_monitor_in = []
    for cc_class in range(num_class):
        aux_sum = 0
        aux_other_sum = 0
        glob_sum = 0
        layer = 1
        temp_res = []
        for i in range(pos.shape[1]):
            for j in range(pos.shape[2]):
                if pos[cc_class, i, j] == 1:
                    temp = []
                    last = w_glob_last['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j]
                    cc = cc_net[cc_class]['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j]
                    # last : 0.026705671101808548 , cc : 0.047083530575037
                    # logging.info(f"last : {last} , cc : {cc}")
                    for p in range(len(cc_net)):
                        temp.append(cc_net[p]['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j] - last)
                        # -0.0013826806098222733
                        # logging.info(f"cc_net[p]['resnet.fc.weight'.format(layer)].cpu().numpy()[i, j] - last {cc_net[p]['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j] - last}")
                    temp = np.array(temp)
                    temp = np.delete(temp, cc_class)
                    '''
                    temp [-0.00134192 -0.00129263 -0.00137705 -0.00145556 -0.00144528 -0.00142462
                     -0.00136255 -0.00128146 -0.00129319 -0.00131153 -0.0012821  -0.00132225
                     -0.00137788 -0.00121276 -0.00133043 -0.0013392  -0.00134192 -0.00138127
                     -0.0014203  -0.00134433 -0.00139391 -0.00136053 -0.00146802 -0.00144278
                     -0.00137475]
                    '''
                    # logging.info(f"temp {temp}")
                    temp_ave = np.sum(temp) / (len(cc_net) - 1)
                    aux_sum += cc - last
                    aux_other_sum += temp_ave

                    glob_temp = (w_glob['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j] - last) * num_users * args.local_bs
                    glob_sum += glob_temp
                    res_temp = (glob_temp - num_samples * temp_ave) / (cc - last - temp_ave)
                    # res_temp :46.55109981762218
                    # logging.info(f"res_temp :{res_temp}")
                    # logging.info(res_temp)
                    if 0 < res_temp < num_samples * 1.5 / num_class:
                        temp_res.append(res_temp)
        if len(temp_res) != 0: 
            res_monitor.append(np.mean(temp_res))
        else:
            res_monitor.append(num_samples / num_class)
        if aux_sum - aux_other_sum == 0:
            res = 0
        else:
            res = (glob_sum - num_samples * aux_other_sum) / (aux_sum - aux_other_sum)
        res_monitor_in.append(res)
        
    res_monitor = np.array(res_monitor)
    res_monitor_in = np.array(res_monitor_in)
    '''
    res_monitor : [19.98055664 16.4290786  18.98354357  8.88386813  7.89114392  6.33521516
         16.01716549 41.00242074 33.50473741 44.12022625 46.92052718 43.69224372
         31.01256811 31.92253839 49.75012894 35.39726263 40.27233251 34.61852436
         26.25715949 32.07317929 36.81282804 31.28020605 33.30634102 30.40849904
         29.90838736 39.29157522]
    '''
    # logging.info(f"res_monitor : {res_monitor}")

    '''
    res_monitor_in : [19.97419716 16.42306871 18.97947312  8.87925804  7.88877808  6.33385228
         16.01280761 55.77191447 46.70213668 59.39216357 64.7192212  64.1265055
         36.777449   36.39454069 83.08814305 42.72009356 50.41043223 44.27209826
         29.25153768 35.85765546 44.23430387 34.27214476 37.59448976 31.63517114
         31.31037619 42.94988532]
    '''
    # logging.info(f"res_monitor_in : {res_monitor_in}")
    return res_monitor, res_monitor_in

def ground_truth_composition(dict_users, idxs_users, num_class, labels):
    res = [0 for i in range(num_class)]
    # 1 2 3
    for idx in idxs_users:
        # idx of data
        for i in dict_users[idx]:
            for j in range(num_class):
                # logging.info(f"labels[i] {labels[i]}")
                # logging.info(f"j {j}")
                temp = np.where(labels[i] == j)
                # position +1
                res[j] += len(temp[0])
                # logging.info(res)
    return res

def cosine_similarity(x, y):
    res = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return res
