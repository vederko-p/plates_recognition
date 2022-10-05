import os
from yaml import load, Loader
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .models.lprnet import build_lprnet
from .models.load_data import CHARS, LPRDataset


def sparse_tuple_for_ctc(t_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(t_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)


def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(int)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)


def train(config: Union[str, dict]):
    if isinstance(config, str):
        with open(config, "r") as f:
            args = load(f, Loader=Loader)

    t_length = 18
    epoch = 0 + args["resume_epoch"]
    loss_val = 0

    if not os.path.exists(args["save_folder"]):
        os.mkdir(args["save_folder"])

    lprnet = build_lprnet(lpr_max_len=args["lpr_max_len"],
                          phase=args["phase_train"],
                          class_num=len(CHARS),
                          dropout_rate=args["dropout_rate"])
    device = torch.device("cuda:0" if args["cuda"] else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args["pretrained_model"]:
        lprnet.load_state_dict(torch.load(args["pretrained_model"]))
        print("Load pretrained model successful!")
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split(".")[-1] == "weight":
                    if "conv" in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode="fan_out")
                    if "bn" in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split(".")[-1] == "bias":
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("Initial net weights successful!")

    optimizer = optim.RMSprop(lprnet.parameters(), lr=args["learning_rate"], alpha = 0.9, eps=1e-08,
                              momentum=args["momentum"], weight_decay=args["weight_decay"])
    # optimizer = optim.Adam(lprnet.parameters(), lr=args["learning_rate"], eps=1e-08,
    #                        betas=(0.9, 0.999), weight_decay=args["weight_decay"])
    train_img_dirs = os.path.expanduser(args["train_img_dirs"])
    test_img_dirs = os.path.expanduser(args["test_img_dirs"])
    train_dataset = LPRDataset(train_img_dirs.split(","), args["img_size"], args["lpr_max_len"])
    test_dataset = LPRDataset(test_img_dirs.split(","), args["img_size"], args["lpr_max_len"])

    epoch_size = len(train_dataset) // args["train_batch_size"]
    max_iter = args["max_epoch"] * epoch_size

    ctc_loss = nn.CTCLoss(blank=len(CHARS) - 1, reduction="mean") # reduction: 'none' | 'mean' | 'sum'

    if args["resume_epoch"] > 0:
        start_iter = args["resume_epoch"] * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(
                DataLoader(train_dataset,
                           args["train_batch_size"],
                           shuffle=True,
                           num_workers=args["num_workers"],
                           collate_fn=collate_fn)
            )
            loss_val = 0
            epoch += 1

        if iteration != 0 and iteration % args["save_interval"] == 0:
            torch.save(lprnet.state_dict(), args["save_folder"] + "LPRNet" + "_iteration_" + repr(iteration) + ".pth")

        if (iteration + 1) % args["test_interval"] == 0:
            get_acc(lprnet, test_dataset, args)
            # lprnet.train() # should be switch to train mode

        # load train data
        images, labels, lengths = next(batch_iterator)
        # labels = np.array([el.numpy() for el in labels]).T
        # print(labels)
        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(t_length, lengths)
        # update lr
        lr = adjust_learning_rate(optimizer, epoch, args["learning_rate"], args["lr_schedule"])

        if args["cuda"]:
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(labels, requires_grad=False).cuda()
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        # forward
        logits = lprnet(images)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        # print(labels.shape)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        # log_probs = log_probs.detach().requires_grad_()
        # print(log_probs.shape)
        # backprop
        optimizer.zero_grad()
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        if loss.item() == np.inf:
            continue
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
        if iteration % 20 == 0:
            print("Epoch:" + repr(epoch) + " || epochiter: " + repr(iteration % epoch_size) + "/" + repr(epoch_size)
                  + "|| Total iter " + repr(iteration) + " || Loss: %.4f||" % (loss.item()) + "LR: %.8f" % (lr))
    # final test
    print("Final test Accuracy:")
    get_acc(lprnet, test_dataset, args)

    # save final parameters
    torch.save(lprnet.state_dict(), args["save_folder"] + "Final_LPRNet_model.pth")


def get_acc(net, datasets, args):
    epoch_size = len(datasets) // args["test_batch_size"]
    batch_iterator = iter(
        DataLoader(datasets,
                   args["test_batch_size"],
                   shuffle=True,
                   num_workers=args["num_workers"],
                   collate_fn=collate_fn)
    )

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0

    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])

        if args["cuda"]:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp + Tn_1 + Tn_2)))
