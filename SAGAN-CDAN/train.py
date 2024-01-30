# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from data_preprocessing import CustomData
from dataset import load_ddi_dataset
from log.train_logger import TrainLogger
from model import SA_DDI
import argparse
from metrics import *
from utils import *
from model import cross_entropy_logits
import warnings
warnings.filterwarnings("ignore")

def val(model, criterion, dataloaders,dataloadert, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for datas,datat in zip(dataloaders,dataloadert):
        head_pairs_s, tail_pairs_s, rel_s, label_s = [d.to(device) for d in datas]
        head_pairs_t, tail_pairs_t, rel_t, label_t = [d.to(device) for d in datat]
        with torch.no_grad():
            pred,x_s,x_t = model((head_pairs_s, tail_pairs_s, rel_s,head_pairs_t, tail_pairs_t, rel_t))
            loss = criterion(pred, label_t)
            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label_t.detach().cpu().numpy())
            running_loss.update(loss.item(), label_t.size(0))

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss, acc, auroc, f1_score, precision, recall, ap,pred_probs

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--n_iter', type=int, default=10, help='number of iterations')
    parser.add_argument('--fold', type=int, default=0, help='[0, 1, 2]')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    args = parser.parse_args()

    params = dict(
        model='SA-DDI',
        #data_root='data/preprocessed/',
        data_root_s='data/preprocessed_s/',
        data_root_t='data/preprocessed_t/',
        save_dir='save',
        dataset='twosides',
        epochs=args.epochs,
        fold=args.fold,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        weight_decay=args.weight_decay
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    save_model = params.get('save_model')
    batch_size = params.get('batch_size')
    data_root_s = params.get('data_root_s')
    data_root_t = params.get('data_root_t')
    #data_root = params.get('data_root')
    data_set = params.get('dataset')
    fold = params.get('fold')
    epochs = params.get('epochs')
    n_iter = params.get('n_iter')
    lr = params.get('lr')
    weight_decay = params.get('weight_decay')
    data_path_s = os.path.join(data_root_s, data_set)
    data_path_t = os.path.join(data_root_t, data_set)
    #data_path = os.path.join(data_root, data_set)
    #train_loader, val_loader, test_loader = load_ddi_dataset(root=data_path, batch_size=batch_size, fold=fold)
    train_loader_s, val_loader_s, test_loader_s = load_ddi_dataset(root=data_path_s, batch_size=batch_size, fold=fold)
    train_loader_t, val_loader_t, test_loader_t = load_ddi_dataset(root=data_path_t, batch_size=batch_size, fold=fold)
    #data = next(iter(train_loader))
    data_s = next(iter(train_loader_s))
    data_t = next(iter(train_loader_t))
    # node_dim = data[0].x.size(-1)
    # edge_dim = data[0].edge_attr.size(-1)
    node_dim_s = data_s[0].x.size(-1)
    edge_dim_s = data_s[0].edge_attr.size(-1)
    node_dim_t = data_t[0].x.size(-1)
    edge_dim_t = data_t[0].edge_attr.size(-1)
    device = torch.device('cuda:0')

    #model = SA_DDI(node_dim, edge_dim, n_iter=n_iter).cuda()
    model = SA_DDI(node_dim_s,node_dim_t, edge_dim_s, n_iter=n_iter).cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))


    running_loss = AverageMeter()
    running_acc = AverageMeter()
#%% s_model----------------------------------------------------------------
    model.train()
    for epoch in range(epochs):
        for datas,datat in zip(train_loader_s,train_loader_t):

            head_pairs_s, tail_pairs_s, rel_s, label_s = [d.to(device) for d in datas]
            head_pairs_t, tail_pairs_t, rel_t, label_t = [d.to(device) for d in datat]
            #label = torch.cat(label_s.unsqueeze(),label_t.unsqueeze())
            pred,x_s,x_t = model((head_pairs_s, tail_pairs_s, rel_s,head_pairs_t, tail_pairs_t, rel_t))

            loss1 = criterion(pred, label_t)
            # loss_cdan_src = criterion(x_s.squeeze(), torch.zeros(512).cuda(0))
            # loss_cdan_tgt = criterion(x_t.squeeze(), torch.ones(512).cuda(0))
            n_src, loss_cdan_src = cross_entropy_logits(x_s,
                                                        torch.zeros(512).cuda(0),
                                                        None)
            n_tgt, loss_cdan_tgt = cross_entropy_logits(x_t,
                                                        torch.ones(512).cuda(0),
                                                        None)
            loss2 = loss_cdan_src + loss_cdan_tgt
            loss = loss1+loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_cls = (torch.sigmoid(pred) > 0.5).detach().cpu().numpy()
            acc = accuracy(label_t.detach().cpu().numpy(), pred_cls)
            running_acc.update(acc)
            running_loss.update(loss.item(), label_t.size(0))

        epoch_loss = running_loss.get_average()
        epoch_acc = running_acc.get_average()
        running_loss.reset()
        running_acc.reset()

        val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap,pred_probs  = val(model, criterion, val_loader_s,val_loader_t, device)

        msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f, val_auroc-%.4f, val_f1_score-%.4f, val_prec-%.4f, val_rec-%.4f, val_ap-%.4f" % (epoch, epoch_loss, epoch_acc, val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap)
        logger.info(msg)

        scheduler.step()

        # if save_model:
        #     msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f" % (epoch, epoch_loss, epoch_acc, val_loss, val_acc)
        #     # del_file(logger.get_model_dir())
        #     save_model_dict(model, logger.get_model_dir(), msg)



# %%
if __name__ == "__main__":
    main()

