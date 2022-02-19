import matplotlib
matplotlib.use('Agg')
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from PIL import Image
import math
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

from utils.sampling import EMNIST_client_imbalance, load_EMNIST_data, EMNIST_client_regenerate, ratio_loss_data
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, Net
from models.Fed import FedAvg, outlier_detect, whole_determination, monitoring, ground_truth_composition, cosine_similarity
from models.test import test_img

import logging


if __name__ == '__main__':
    # parse args
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    args = args_parser()
    logging.info(args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.device = torch.device('cpu')
    logging.info(args.device)

    # load dataset and split users
    if args.dataset == 'femnist':

        trans_femnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train, label_train, dataset_test, label_test, w_train, w_test = load_EMNIST_data('../emnist-letters.mat', verbose=True, standarized=False)
        if args.iid != 0:
            dict_users = EMNIST_client_imbalance(dataset_train, label_train, w_train, 100, [0, 1, 2, 3, 4, 5, 6], 0.05)
        else:
            dict_users = EMNIST_client_regenerate(dataset_train, label_train, w_train, 100)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    train_data_num = [len(dict_users[r]) for r in range(args.num_users)]

    logging.info(f"Client train quantity dict : {train_data_num}")
    logging.info(f"All sample : {sum(train_data_num)}")

    net_cls_counts = {}

    for net_i, dataidx in dict_users.items():
        unq, unq_cnt = np.unique(label_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    # logging.info('Data statistics: %s' % str(net_cls_counts))

    # build model
    if args.model == 'cnn' and args.dataset == 'femnist':
        net_glob = Net().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    # logging.info(net_glob)

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    sim_1 = []
    selection = []
    labels = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    ratio = None
    val_acc_list, net_list = [], []
    dict_ratio = ratio_loss_data(dataset_train, label_train, w_train, 26, args)
    # logging.info(f"dict_ratio {dict_ratio}")

    for iter in range(args.epochs):
        if iter > 135:
            args.lr = 0.01
        w_locals, loss_locals, ac_locals, num_samples = [], [], [], []
        m = max(int(args.frac * args.num_users), 1)
        np.random.seed(100)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        logging.info(f"Selected clients : {idxs_users}")

        pro_ground_truth = ground_truth_composition(dict_users, idxs_users, 26, label_train)
        logging.info(f"ground_truth_composition : {pro_ground_truth}")
        logging.info(f"label_train.shape : {label_train.shape}")

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, label=label_train, idxs=dict_users[idx], alpha=ratio, size_average=True)
            w, loss, ac= local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            ac_locals.append(copy.deepcopy(ac))
            num_samples.append(len(dict_users[idx]))
            logging.info(f"Local train {idx}")

        # monitor
        cc_net, cc_loss = [], []
        aux_class = [i for i in range(26)]
        for i in aux_class:
            cc_local = LocalUpdate(args=args, dataset=dataset_train, label=label_train, idxs=dict_ratio[i], alpha=None, size_average=True)
            cc_w, cc_lo, cc_ac = cc_local.train(net=copy.deepcopy(net_glob).to(args.device))
            cc_net.append(copy.deepcopy(cc_w))
            cc_loss.append(copy.deepcopy(cc_lo))
            logging.info(f"Local aux {i}")
        pos = outlier_detect(w_glob, cc_net, iter)
        # logging.info(f"Outlier pos list: {pos}")

        # labelling process and updating global model
        w_glob_last = copy.deepcopy(w_glob)
        w_glob = FedAvg(w_locals)
        
        num_sample = np.sum(num_samples)
        pro_res_1, pro_res_2 = monitoring(cc_net, pos, w_glob_last, w_glob, 26, m, num_sample, args)
        # logging.info(pro_res_1, '\n', pro_res_2)
        logging.info(f"pro_res_1 : {pro_res_1}")
        logging.info(f"pro_ground_truth : {pro_ground_truth}")
        logging.info("cs_1: {:.4f}, cs_2: {:.4f}".format(cosine_similarity(pro_res_1, pro_ground_truth), cosine_similarity(pro_res_2, pro_ground_truth)))
        sim_1.append(cosine_similarity(pro_res_1, pro_ground_truth))
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # logging.info loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        ac_avg = sum(ac_locals) / len(ac_locals)
        logging.info('Round {:3d}, Average loss {:.3f}, Accuracy {:.3f}\n'.format(iter, loss_avg, ac_avg))
        loss_train.append(loss_avg)

    # testing
    np.savetxt('{}-num-{}.csv'.format(args.dataset, int(args.frac * args.num_users)), sim_1, delimiter=',')

    net_glob.eval()
    logging.info('FL(femnist, mismatch 4): {}, 100:1, [0, 1, 2], {} eps, {} local eps'.format(args.loss, args.epochs, args.local_ep))

    acc_test, loss_test = test_img(net_glob, dataset_test, label_test, args)
    logging.info("Testing accuracy: {:.2f}".format(acc_test))

