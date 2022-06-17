import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
import logging
from datetime import datetime
import os
import numpy as np, random

from sklearn.metrics import f1_score, accuracy_score, classification_report

from model import MaskedNLLLoss, LSTMModel, GRUModel, DialogRNNModel, ResidualGCNEC

from torchtext import data
from torchtext.data import BucketIterator, Pipeline
from keras.utils import np_utils
#from keras.utils.np_utils import to_categorical

import spacy

spacy_en = spacy.load('en')

label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}

# We use seed = 100 for reproduction of the results reported in the paper.
seed = 100

def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tokenizer(text):
    return [token.text for token in spacy_en.tokenizer(text)]


def convert_token(token, *args):
    return token - 1


def get_E2E_loaders(path, valid=0.1, batch_size=32):
    utterance = data.Field(tokenize=tokenizer, lower=True)
    label = data.Field(sequential=False, postprocessing=Pipeline(convert_token=convert_token))
    id = data.Field(use_vocab=False, sequential=False)
    fields = [('id', id),
              ('turn1', utterance),
              ('turn2', utterance),
              ('turn3', utterance),
              ('label', label)]

    train = data.TabularDataset('{}/train.txt'.format(path),
                                format='tsv',
                                fields=fields,
                                skip_header=True)
    valid = data.TabularDataset('{}/valid.txt'.format(path),
                                format='tsv',
                                fields=fields,
                                skip_header=True)

    test = data.TabularDataset('{}/test.txt'.format(path),
                               format='tsv',
                               fields=fields,
                               skip_header=True)
    #vectors = vocab.Vectors(name='emojiplusglove.txt', cache='/media/backup/nlp-cic/DialogueRNN/')
    #utterance.build_vocab(train, valid, test, vectors=vectors)
    utterance.build_vocab(train, valid, test, vectors='glove.840B.300d'.format(path))
    label.build_vocab(train)
    train_iter = BucketIterator(train,
                                train=True,
                                batch_size=batch_size,
                                sort_key=lambda x: len(x.turn3),
                                device=torch.device(0))
    valid_iter = BucketIterator(valid,
                                batch_size=batch_size,
                                sort_key=lambda x: len(x.turn3),
                                device=torch.device(0))
    test_iter = BucketIterator(test,
                               batch_size=batch_size,
                               sort_key=lambda x: len(x.turn3),
                               device=torch.device(0))
    return train_iter, valid_iter, test_iter, \
           utterance.vocab.vectors if not args.cuda else utterance.vocab.vectors.cuda(), \
           label.vocab.itos

def train_or_eval_graph_model(model, loss_function, loss_function_edge, dataloader, epoch, cuda, optimizer=None,
                              train=False):
    losses, preds, labels, masks = [], [], [], []
    scores = []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    # if torch.cuda.is_available():
    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        log_prob, e_i, e_n, e_t, e_l = model(data, True)  # batch*3, n_classes

        lp_ = []
        for j in range(log_prob.size(0) // 3):
            lp_.append(log_prob[j*3+2, :])

        lp_ = torch.stack(lp_, dim=0)  # batch, n_classes

        labels_ = data.label  # batch
        loss = loss_function(lp_, labels_)

        e_identity = torch.zeros(e_i.size(1))
        e_identity_label = torch.zeros(e_i.size(1))

        for i in range(e_i.size(1)):
            if e_i[0, i] == e_i[1, i]:
                e_identity[i] = e_n[i]
                e_identity_label[i] = 1

        loss_edge = loss_function_edge(e_identity.cuda(), e_identity_label.cuda())

        lambda_edge_loss = 0.001

        loss = loss + lambda_edge_loss * loss_edge

        ei = torch.cat([ei, e_i], dim=1)
        et = torch.cat([et, e_t])
        en = torch.cat([en, e_n])
        el += e_l

        preds.append(torch.argmax(lp_, 1).cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    pass
                    # writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    _, _, _, avg_fscore_micro = get_metrics(labels, preds)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    print('F-Score: {}'.format(avg_fscore))

    print(classification_report(labels, preds, digits=4))

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, avg_fscore_micro, ei, et, en, el

def get_metrics(discretePredictions, ground, n_classes=4):
    discretePredictions = np_utils.to_categorical(discretePredictions, 4)
    ground = np_utils.to_categorical(ground, 4)
    truePositives = np.sum(discretePredictions * ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    accuracy = np.mean(discretePredictions == ground)
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, n_classes):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision) / (macroPrecision + macroRecall) if (
                                                                                                 macroPrecision + macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (
    macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    print(
        "Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall) if (
                                                                                                 microPrecision + microRecall) > 0 else 0
    # -----------------------------------------------------

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (
    accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


if __name__ == '__main__':
    model_path = '../pretrained_models/RGCN-EC.pkl'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True,
                        help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=True,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.00005, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')

    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available() and not args.no_cuda

    batch_size = args.batch_size
    n_classes = 4
    n_epochs = args.epochs
    cuda = args.cuda

    D_emb = 300
    D_m = 200
    D_r = 328  # 200
    D_g = 150
    D_p = 150
    D_e = 200
    D_h = 100
    graph_h = 100

    D_a = 100  # concat attention

    lambda_edge = 0.001

    loss_weights = torch.FloatTensor([2.0, 1.0, 1.0, 1.0])

    train_loader, valid_loader, test_loader, embeddings, id2label = \
        get_E2E_loaders('../datasets/EC',
                        valid=0.1,
                        batch_size=batch_size)

    if args.graph_model:
        seed_everything()
        model = ResidualGCNEC(args.base_model,
                                 D_emb, D_m, D_r, D_g, D_p, D_e, D_h, D_a, embeddings, graph_h,
                                 n_speakers=2,
                                 max_seq_len=110,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda)
        name = 'Graph'

    else:
        if args.base_model == 'DialogRNN':
            model = DialogRNNModel(D_m, D_g, D_p, D_e, D_h, D_a,
                                   n_classes=n_classes,
                                   listener_state=args.active_listener,
                                   context_attention=args.attention,
                                   dropout_rec=args.rec_dropout,
                                   dropout=args.dropout)


        elif args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h,
                             n_classes=n_classes,
                             dropout=args.dropout)


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h,
                              n_classes=n_classes,
                              dropout=args.dropout)

        else:
            raise NotImplementedError

        name = 'Base'

    if cuda:
        model.cuda()

    if args.class_weight:
        if args.graph_model:
            loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        if args.graph_model:
            loss_function = nn.NLLLoss()
        else:
            loss_function = MaskedNLLLoss()

    loss_function_edge = nn.L1Loss(reduction='sum')

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if args.graph_model:
        test_loss, test_acc, test_label, test_pred, test_fscore, test_fscore_micro, _, _, _, _ = train_or_eval_graph_model(
            model,
            loss_function,
            loss_function_edge,
            test_loader,
            0, cuda)
