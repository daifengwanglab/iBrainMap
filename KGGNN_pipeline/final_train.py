import torch
torch.backends.cudnn.benchmark=False
import pickle
import os
import gc
import argparse
import warnings
import random
import json

from collections import Counter
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

#from pytorchtools import EarlyStopping
from torch_geometric.loader import DataLoader
import sklearn.metrics as skm
import iBrainMap_data_utils as idu
from iBrainMap_KGGNN_model import KG_GNN

def set_seed(seed: int = 42) -> None:
    """ Function to set seed """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def get_optimizer(opt_name, model, learn_rate):
    """ Function to define the optimizer """
    if opt_name =='Adagra':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learn_rate)
    if opt_name =='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    if opt_name =='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    if opt_name =='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)
    if opt_name =='Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=learn_rate)
    if opt_name =='ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=learn_rate)
    if opt_name =='LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learn_rate)
    if opt_name =='NAdam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=learn_rate)
    if opt_name =='RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=learn_rate)
    if opt_name =='RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learn_rate)
    if opt_name =='Rprop':
        optimizer = torch.optim.Rprop(model.parameters(), lr=learn_rate)
    return optimizer

def get_classification_performance(y_true, y_pred, y_pred_score, num_classes):
    """Function to return the acc, bacc, auprc, and auc"""
    auc = None
    if num_classes == 2:
        auc = skm.roc_auc_score(y_true, y_pred_score)
    else:
        auc = skm.roc_auc_score(y_true, y_pred_score, average="weighted", multi_class="ovr")

    auprc = skm.average_precision_score(y_true, y_pred_score, average='weighted')
    acc = skm.accuracy_score(y_true, y_pred)
    bacc = skm.balanced_accuracy_score(y_true, y_pred)
    return acc, bacc, auc, auprc


def train_step(model, data_dl, num_classes, optimizer, scheduler):
    """ Function to train an epoch """
    total_loss = 0
    y_pred, y_true, y_pred_score = [], [], []
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for data in data_dl:
        data = data.to(device)
        out, _, _ = model(data.x, data.edge_index, data.batch, data.edge_attr,
                          return_attention_weights=True)

        data.y = data.y.to(torch.int64)
        # print(data.y)
        pred_loss = criterion(out, data.y)

        total_loss += pred_loss.detach().cpu().numpy()
        y_true.extend(data.y.detach().cpu().numpy())
        pred_score = torch.softmax(out, dim=1).detach().cpu().numpy()

        if num_classes==2:
            y_pred_score.extend(pred_score[:, 1])
        elif num_classes>2:
            y_pred_score.extend(pred_score)

        y_pred.extend(out.argmax(1).detach().cpu().numpy())

        if optimizer is not None:
            optimizer.zero_grad()
            pred_loss.backward()
            optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return total_loss/len(data_dl), y_true, y_pred, np.asarray(y_pred_score)

def predict(model, data_dl, num_classes):
    """ Function to train an epoch """
    with torch.no_grad():
        model.eval()

        y_pred, y_true, y_pred_score = [], [], []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for data in data_dl:
            data = data.to(device)
            out, _, _ = model(data.x, data.edge_index, data.batch,
                              data.edge_attr, return_attention_weights=True)
            data.y = data.y.to(torch.int64)
            y_true.extend(data.y.detach().cpu().numpy())
            pred_score = torch.softmax(out, dim=1).detach().cpu().numpy()

            if num_classes==2:
                y_pred_score.extend(pred_score[:, 1])
            elif num_classes>2:
                y_pred_score.extend(pred_score)
            y_pred.extend(out.argmax(1).detach().cpu().numpy())


            #print(attn_mat[0:5, :])
            del (data)
            gc.collect()
            torch.cuda.empty_cache()
            #info = torch.cuda.mem_get_info()
            #print('Total CUDA memory free: %.2f mb'%(info[0]/1024**2))

    return y_true, y_pred, np.asarray(y_pred_score)
    # return y_true, y_pred, np.asarray(y_pred_score)


def fit(epochs, stagnant_val, model, optimizer, scheduler, tr_loader, te_loader, num_classes):
    """ Function for trinaing the model and selecting the best performance """
    max_tr_bacc, max_val_bacc = 0, 0
    max_tr_acc, max_val_acc = 0, 0
    max_tr_auprc, max_val_auprc  = 0, 0
    max_tr_auc, max_val_auc = 0, 0

    # max_val_loss = 0.0
    stagnant = 0

    best_model = None
    best_val_true, best_val_pred_score = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        tr_loss, tr_true, tr_pred, tr_score = train_step(model, tr_loader, num_classes,
                                                         optimizer, scheduler)

        # Validation phase
        with torch.no_grad():
            model.eval()
            val_loss, val_true, val_pred, val_score = train_step(model, te_loader,
                                                                 num_classes, None, None)

            # Performance evaluation
            tr_acc, tr_bacc, tr_auc, tr_prc = get_classification_performance(tr_true, tr_pred,
                                                                             tr_score, num_classes)
            val_acc, val_bacc, val_auc, val_prc = get_classification_performance(val_true,
                                                                                 val_pred,
                                                                                 val_score,
                                                                                 num_classes)

            print("\n*** Epoch = %d ***"%(epoch))
            print("Train: Loss- %.4f, ACC- %.4f, BACC- %.4f, AUC- %.4f, AUPRC- %.4f"%(tr_loss,
                                                                                      tr_acc,
                                                                                      tr_bacc,
                                                                                      tr_auc,
                                                                                      tr_prc))
            print(skm.confusion_matrix(tr_true, tr_pred))

            print("Val: Loss- %.4f, ACC- %.4f, BACC- %.4f, AUC- %.4f, AUPRC- %.4f"%(val_loss,
                                                                                    val_acc,
                                                                                    val_bacc,
                                                                                    val_auc,
                                                                                    val_prc))
            print(skm.confusion_matrix(val_true, val_pred))

            if epoch == 0:
#                max_val_loss = val_loss
                max_tr_acc, max_val_acc = tr_acc, val_acc
                max_tr_bacc, max_val_bacc = tr_bacc, val_bacc
                max_tr_auprc, max_val_auprc = tr_prc, val_prc
                max_tr_auc, max_val_auc = tr_auc, val_auc
                best_model = model
                best_val_true, best_val_pred_score = val_true, val_score
                stagnant = 0
                print(f'Stagnant after this epoch: {stagnant}')
            else:
                stagnant += 1
                if val_bacc >= max_val_bacc and tr_bacc >= val_bacc: # and (val_loss < max_val_loss):
                    if (tr_bacc >= 0.95) and ((tr_bacc - val_bacc) >= 0.15):
                        stagnant += 1
                    else:
                        max_tr_acc, max_val_acc = tr_acc, val_acc
                        max_tr_bacc, max_val_bacc = tr_bacc, val_bacc
                        max_tr_auprc, max_val_auprc = tr_prc, val_prc
                        max_tr_auc, max_val_auc = tr_auc, val_auc
                        best_model = model
                        best_val_true, best_val_pred_score = val_true, val_score

                    stagnant = 0
                    print(f'Stagnant after this epoch: {stagnant}')
                    continue
                print(f'Stagnant after this epoch: {stagnant}')
            if stagnant >= stagnant_val and epoch > 13:
                break
    tr_dict = {'acc': max_tr_acc, 'bacc': max_tr_bacc, 'auc': max_tr_auc, 'auprc': max_tr_auprc}

    val_dict = {'acc': max_val_acc, 'bacc': max_val_bacc, 'auc': max_val_auc,
                'auprc': max_val_auprc, 'truth': best_val_true,
                'pred_scores':best_val_pred_score}

    return best_model, tr_dict, val_dict


def train(args):
    """ Function to run cross validation modelling"""
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    ### ---------------------- Step 1: Load data --------------------- ###
    # Load all the graphs
    graphs, sample_ids, _ = idu.get_graphs(args.data_dir, args.phenotype_file, args.id_column,
                                           args.phenotype_column, args.edge_attr_comb)

    # Split the data into training and heldout based on heldout sample IDS
    test_samples = None
    if args.heldout_sampleID_file != 'None':
        test_samples = pd.read_csv(args.heldout_sampleID_file)
        test_samples = test_samples.iloc[:, 0].tolist()

    tr_graphs, te_graphs, id_train, id_test = idu.split_train_heldout_graphs(graphs, sample_ids,
                                                                             test_samples)

    # Class labels
    y_tr = [tr_graphs[i].y.item() for i in range(len(tr_graphs))]
    y_te = [te_graphs[i].y.item() for i in range(len(te_graphs))]

    # Get train and test sample ids
    print(len(id_train), len(id_test))

    ### -----------------Step 2: Generate sub-graphs ------------------ ###
    args.num_classes = len(set(y_tr))

    # Generate subgraphs for Training and testing samples
    X_tr_sub, X_te_sub = [], []
    if args.num_classes == 2:
        X_tr_sub, _ = idu.get_subgraph(tr_graphs, id_train, args.num_subgraph, y_tr)
        X_te_sub, _ = idu.get_subgraph(te_graphs, id_test, args.num_subgraph, y_tr)
    else:
        X_tr_sub, _ = idu.get_subgraph_multiclass(tr_graphs, id_train, args.num_subgraph,
                                                  y_tr, balanced=True)
        X_te_sub, _ = idu.get_subgraph_multiclass(te_graphs, id_test, args.num_subgraph, y_te)

    X_tr_sub.extend(tr_graphs)
    y_tr_sub = [X_tr_sub[i].y.item() for i in range(len(X_tr_sub))]

    X_te_sub.extend(te_graphs)
    y_te_sub = [X_te_sub[i].y.item() for i in range(len(X_te_sub))]

    print('Train samples with original grpahs: ', len(tr_graphs), Counter(y_tr))
    print('Train samples with subgrpahs: ', len(X_tr_sub), Counter(y_tr_sub))
    print('Validation samples with original grpahs: ', len(te_graphs), Counter(y_te))
    print('Validation samples with subgrpahs: ', len(X_te_sub), Counter(y_te_sub))

    # Create data loaders
    tr_sub_loader = DataLoader(X_tr_sub, batch_size=args.batch_size, shuffle=False)
    te_sub_loader = DataLoader(X_te_sub, batch_size=args.batch_size, shuffle=False)

    ### -----------------Step 3: Model creation------------------ ###
    # Get model parameters
    input_feat_len = tr_graphs[0].num_node_features
    num_gat_nodes = [int(x) for x in args.num_gat_nodes.split(',')]
    num_fcn_nodes = [int(x) for x in args.num_fcn_nodes.split(',')]

    model = KG_GNN(input_feat_len, num_gat_nodes, args.num_heads, num_fcn_nodes,
                    args.num_classes, args.dropout, need_layer_norm=args.need_layer_norm,
                    need_batch_norm=args.need_layer_norm, need_attn_concat=False)
    # model = GCN(input_feat_len, num_gat_nodes, args.num_classes, args.dropout)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(model)

    # Define the loss function snd initialize optimizer
    opt = get_optimizer(args.opt, model, args.learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

    ### -----------------Step 4: Train model ------------------ ###
    mdl, perf_tr, perf_te = fit(args.epochs, args.stagnant, model, opt, scheduler,
                                tr_sub_loader, te_sub_loader, args.num_classes)

    info = torch.cuda.mem_get_info()
    # print('Total CUDA memory allocated: %.2f mb'%(info[1]/1024**2))
    # print('Total CUDA memory free: %.2f mb'%(info[0]/1024**2))

    del (tr_sub_loader, te_sub_loader, X_tr_sub, X_te_sub)
    gc.collect()
    #torch.cuda.empty_cache()
    print('Deleted unused variables')

    info = torch.cuda.mem_get_info()
    print('Total CUDA memory free: %.2f mb\n\n'%(info[0]/1024**2))

    ### ---------------- Step 5: Test on Whole Graphs ----------------- ###
    tr_loader = DataLoader(tr_graphs, batch_size=1, shuffle=False)
    te_loader = DataLoader(te_graphs, batch_size=1, shuffle=False)

    tro_true, tro_pred, tro_pred_score = predict(mdl, tr_loader, args.num_classes)
    _, tro_bacc, tro_auc, tro_auprc = get_classification_performance(tro_true, tro_pred,
                                                                     tro_pred_score,
                                                                     args.num_classes)

    teo_true, teo_pred, teo_pred_score= predict(mdl, te_loader, args.num_classes)
    _, teo_bacc, teo_auc, teo_auprc = get_classification_performance(teo_true, teo_pred,
                                                                     teo_pred_score,
                                                                     args.num_classes)

    ### ----------------- Print and save results ---------------- ###
    # Save model
    if args.model_name == '':
        args.model_name = 'best_mdl'
    torch.save(mdl.state_dict(), os.path.join(args.save, args.model_name + '.pth'))

    # Save prediction scores and true labels of the Training original graphs for ROC curve plotting
    roc_plot_results = {}
    roc_plot_results['samples'] = id_train
    roc_plot_results['tr_orig_truth'] = tro_true
    roc_plot_results['tr_orig_pred_scores'] = tro_pred_score

    with open(os.path.join(args.save, args.model_name+'_train_orig.pkl'), 'wb') as f:
        pickle.dump(roc_plot_results, f)

    # Save prediction scores and true labels of the test original graphs for ROC curve plotting
    roc_plot_results = {}
    roc_plot_results['samples'] = id_test
    roc_plot_results['te_orig_truth'] = teo_true
    roc_plot_results['te_orig_pred_scores'] = teo_pred_score

    with open(os.path.join(args.save, args.model_name+'_test_orig.pkl'), 'wb') as f:
        pickle.dump(roc_plot_results, f)

    out_file = 'test_' + args.phenotype_column + ' _perf_record.txt'
    header_str = "Model,Train_Sub_BACC,Train_Sub_AUPRC,Train_Sub_AUC"
    header_str += ",Val_Sub_BACC,Val_Sub_AUPRC,Val_Sub_AUC"
    header_str += ",Train_Orig_BACC,Train_Orig_AUPRC,Train_Orig_AUC"
    header_str += ",Val_Orig_BACC,Val_Orig_AUPRC,Val_Orig_AUC\n"

    if not os.path.exists(out_file):
        with open(out_file, 'w') as write_fl:
            write_fl.write(header_str)
            write_fl.close()

    write_str = "%s,%.5f,%.5f,%.5f"%(args.verbose, perf_tr['bacc'], perf_tr['auprc'], perf_tr['auc'])
    write_str += ",%.5f,%.5f,%.5f"%(perf_te['bacc'], perf_te['auprc'], perf_te['auc'])
    write_str += ",%.5f,%.5f,%.5f"%(tro_bacc, tro_auprc, tro_auc)
    write_str += ",%.5f,%.5f,%.5f"%(teo_bacc, teo_auprc, teo_auc)

    with open(out_file, 'a') as write_fl:
        write_fl.write(write_str)
        write_fl.close()

    # Print the performance
    print("\nKGGNN model training complete")
    print('Train SUB BACC = ', perf_tr['bacc'])
    print('Train SUB AUPRC = ', perf_tr['auprc'])
    print('Train SUB AUC = ', perf_tr['auc'])
    print("")

    print('Val SUB BACC = ', perf_te['bacc'])
    print('Val SUB AUPRC = ', perf_te['auprc'])
    print('Val SUB AUC = ', perf_te['auc'])
    print("")

    print('Train ORIG BACC = ', tro_bacc)
    print('Train ORIG AUPRC = ', tro_auprc)
    print('Train ORIG AUC = ', tro_auc)
    print("")

    print('VAL ORIG BACC = ', teo_bacc)
    print('VAL ORIG AUPRC = ', teo_auprc)
    print('VAL ORIG AUC = ', teo_auc)
    print("")

    perf_dict = {'tr_orig_bacc': tro_bacc, 'tr_orig_auprc': tro_auprc, 'tr_orig_auc': tro_auc,
                 'te_orig_bacc': teo_bacc, 'te_orig_auprc': teo_auprc, 'te_orig_auc': teo_auc}

    if args.return_results:
        return perf_dict


def main():
    """ Main method """
    
    parser = argparse.ArgumentParser()
    print('test')
    # Input
    parser.add_argument('--data_dir', type=str, help='Path to the folder containing all individual graphs',
                        default='./demo/step2/')
    parser.add_argument('--heldout_sampleID_file', type=str, help='Path to the heldout sampleID file',
                        default='')
    parser.add_argument('--phenotype_file', type=str, help='Path to the phenotype file',
                        default='./demo/step2/metadata.csv')
    parser.add_argument('--id_column', type=str, help='specify which column to use for ID',
                        default='sampleid')
    parser.add_argument('--phenotype_column', type=str, help='specify which column to use for labels',
                        default='labels')
    parser.add_argument('--save', type=str, help="path to save model",
                        default="./demo/step2/test/")
    parser.add_argument('--config_file', type=str, help="path to KGGNN config files",
                        default="./demo/step2/config.txt")
    parser.add_argument('--verbose', type=str, help="Name to be given to the analysis",
                        default="test")
    inp_args = parser.parse_args()
    print(inp_args)

    print("\n test2")
    # Opening JSON file
    with open(inp_args.config_file) as json_file:
        data = json.load(json_file)
    
    for k, v in data.items():
        print(k, v)
        parser.add_argument('--' + k, default=v)

    parser.add_argument('--model_name', type=str, help="path to KGGNN config files", default='')
    inp_args = parser.parse_args()
    print(inp_args)

    perf = train(inp_args)
    
    with open(inp_args.save + inp_args.verbose + '_test_results.txt', 'w') as f:
        for k in perf.keys():
            wr_str = k + ': ' + str(perf[k])
            f.writelines(wr_str + '\n')

if __name__ == '__main__':
    main()
