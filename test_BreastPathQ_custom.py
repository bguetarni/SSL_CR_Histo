import argparse, os, random, math, glob
import numpy as np
import pandas
from PIL import Image
from tqdm import tqdm
from sklearn import metrics as sklearn_metrics

import torch.backends.cudnn as cudnn
import torch
from torchvision import transforms
import einops

import models.net as net


def metrics_fn(args, Y_true, Y_pred):
    """
    args : script arguments
    Y_true (n_samples, n_classes) : array of probabilities prediction
    Y_pred (n_samples, n_classes) : array of true class as one-hot index
    """

    # cross-entropy error
    error = sklearn_metrics.log_loss(Y_true, Y_pred)

    # ROC AUC (per class)
    auc = dict()
    for i in range(Y_true.shape[1]):
        # select class one-hot values
        ytrue = Y_true[:,i]

        # transform probabilities from [0.5,1] to [0,1]
        # probabilities in [0,0.5] are clipped to 0
        ypred = np.clip(Y_pred[:,i], 0.5, 1) * 2 - 1
        auc_score = sklearn_metrics.roc_auc_score(ytrue, ypred)
        auc.update({i: auc_score})
    
    # convert to one-hot index
    Y_true_label = np.argmax(Y_true, axis=-1)
    Y_pred_label = np.argmax(Y_pred, axis=-1)
    
    # global metrics
    TP = sklearn_metrics.accuracy_score(Y_true_label, Y_pred_label, normalize=False)
    accuracy = sklearn_metrics.accuracy_score(Y_true_label, Y_pred_label)
    micro_Fscore = sklearn_metrics.f1_score(Y_true_label, Y_pred_label, average='micro')
    macro_Fscore = sklearn_metrics.f1_score(Y_true_label, Y_pred_label, average='macro')
    weighted_Fscore = sklearn_metrics.f1_score(Y_true_label, Y_pred_label, average='weighted')

    # compile metrics in dict
    metrics_ = dict(error=error, TP=TP, accuracy=accuracy, micro_Fscore=micro_Fscore, macro_Fscore=macro_Fscore, weighted_Fscore=weighted_Fscore)
    
    # confusion matrix for each class
    multiclass_cm = sklearn_metrics.multilabel_confusion_matrix(Y_true_label, Y_pred_label)

    # computes binary metrics for each class (one versus all)
    for k, i in TRAIN_PARAMS['class_to_label'][args.dataset].items():
        
        # statistics from sklearn confusion matrix
        tn, fp, fn, tp = multiclass_cm[i].ravel()

        # metrics
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fnr = fn / (fn + tp)
        fscore = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        
        metrics_.update({
            "{}_auc".format(k): auc[i],
            "{}_precision".format(k): precision,
            "{}_recall".format(k): recall,
            "{}_fscore".format(k): fscore,
            "{}_fnr".format(k): fnr,
            })

    return metrics_

def parse_args():

    parser = argparse.ArgumentParser('Argument for BreastPathQ: Supervised Fine-Tuning/Evaluation')

    parser.add_argument('--dataset', type=str, required=True, choices=["chulille", "dlbclmorph", "bci"])
    parser.add_argument('--test_image_pth', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True, help='path to load model')
    parser.add_argument('--fold', type=str, default=None, help='fold to use as test')
    parser.add_argument('--num_classes', type=int, required=True, help='# of classes.')
    parser.add_argument('--image_size', required=True, type=int, help='patch size width 256')
    parser.add_argument('--labels', type=str, default=None, help='path to the label CSV files')
    parser.add_argument('--output', type=str, required=True, help='path to the CSV file to save results')
    parser.add_argument('--gpu', type=str, default="", help='GPU id to use.')
    
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use.')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training.')

    # model definition
    parser.add_argument('--model', type=str, default='resnet18', help='choice of network architecture.')
    parser.add_argument('--mode', type=str, default='fine-tuning', help='fine-tuning/evaluation')
    parser.add_argument('--modules', type=int, default=0, help='which modules to freeze for fine-tuning the pretrained model. (full-finetune(0), fine-tune only FC layer (60) - Resnet18')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size.')

    args = parser.parse_args()
    return args

def load_data(args):
    
    if args.dataset in ["chulille", "dlbclmorph"]:
        labels = pandas.read_csv(args.labels).set_index('patient_id')['label'].to_dict()

    data = []
    if args.dataset == "chulille":
        pass #TODO
    elif  args.dataset == "dlbclmorph":
        for fold in os.listdir(args.test_image_pth):
            if fold != args.fold:
                continue

            for p in os.listdir(os.path.join(args.test_image_pth, fold)):
                y = labels[int(p)]
                label = TRAIN_PARAMS['class_to_label'][args.dataset][y]
                associate_label = lambda i : (i, label)
                list_of_patches = glob.glob(os.path.join(args.test_image_pth, fold, p, '*.png'))
                data.extend(list(map(associate_label, list_of_patches)))
    elif  args.dataset == "bci":
        for img in glob.glob(os.path.join(args.test_image_pth, "test", "*.png")):
            y = os.path.splitext(os.path.split(img)[1])[0].split('_')[-1]
            data.append((img, TRAIN_PARAMS['class_to_label'][args.dataset][y]))
    
    return data

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = net.TripletNet_Finetune(args.model)
    classifier = net.FinetuneResNet(args.num_classes)

    model = torch.nn.DataParallel(model)
    classifier = torch.nn.DataParallel(classifier)

    # Load fine-tuned model
    state = torch.load(os.path.join(args.model_path, "best_CR_trained_model.pt"), map_location='cpu')
    model.load_state_dict(state['model_student'])
    classifier.load_state_dict(state['classifier_student'])

    model.eval()
    classifier.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

    # Test set
    test_transforms = transforms.Compose([transforms.Resize(size=args.image_size)])

    data = load_data(args)

    result = []
    for batch in tqdm(np.array_split(data, math.ceil(len(data) / args.batch_size)), ncols=50):
        
        x, y = zip(*batch)
        open_patches = lambda i : Image.open(i).convert('RGB').reduce(2)
        x = map(open_patches, x)
        x = map(test_transforms, x)
        x = map(np.array, x)
        x = map(torch.from_numpy, x)
        x = torch.stack(list(x), dim=0)
        x = einops.rearrange(x, 'n h w c -> n c h w')
    
        input = x.to(device=device, dtype=torch.float)
        with torch.no_grad():
            feats = model(input)
            output = classifier(feats)
            y_pred = output.to('cpu').numpy()
        
        y = torch.tensor(list(map(int, y)))
        y = torch.nn.functional.one_hot(y, num_classes=len(TRAIN_PARAMS['class_to_label'][args.dataset]))
        y = np.array(y)
        
        result.append((y, y_pred))

    # calculates metrics on patients
    y_true, y_pred = zip(*result)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    patients_metrics = metrics_fn(args, y_true, y_pred)

    # add model result
    df = pandas.DataFrame([patients_metrics])

    # round floats to 2 decimals
    df = df.round(decimals=2)

    # save results in a CSV file
    base, model_name = os.path.split(args.model_path)
    output_path = os.path.join(args.output, model_name+".csv")
    df.to_csv(output_path, sep=';')

if __name__ == "__main__":

    global TRAIN_PARAMS
    TRAIN_PARAMS = dict(
        # dictionnar to convert class name to label
        class_to_label = {
            "chulille": {'ABC': 1, 'GCB': 0},
            "dlbclmorph": {'NGC': 1, 'GC': 0},
            "bci": {'0': 0, '1+': 1, '2+': 2, '3+': 3},
        },
    )

    args = parse_args()
    print(vars(args))

    # Force the pytorch to create context on the specific device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

    # Main function
    main(args)
