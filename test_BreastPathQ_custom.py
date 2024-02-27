import os, argparse, random, re
import numpy as np
import pandas as pd
import tqdm
import torch.backends.cudnn as cudnn
from sklearn import metrics
import openslide

import torch
from torchvision import transforms
import einops

import models.net as net

slide_class_idx = {'ABC': 1, 'GCB': 0}

def scale_coordinates(wsi, p, source_level, target_level):

    if not isinstance(p, np.ndarray):
        p = np.asarray(p).squeeze()

    assert p.ndim < 3 and p.shape[-1] == 2, 'coordinates must be a single point or an array of 2D-cooridnates'

    # source level dimensions
    source_w, source_h = wsi.level_dimensions[source_level]
    
    # target level dimensions
    target_w, target_h = wsi.level_dimensions[target_level]
    
    # scale coordinates
    p = np.array(p)*(target_w/source_w, target_h/source_h)
    
    # round to int64
    return np.floor(p).astype('int64')

def metrics_fn(Y_true, Y_pred, class_idx):
    """
    Y_true (n_samples, n_classes) : array of probabilities prediction
    Y_pred (n_samples, n_classes) : array of true class as one-hot index
    class_idx : dict of class index
    """

    # cross-entropy error
    error = metrics.log_loss(Y_true, Y_pred)
    
    # convert to one-hot index
    Y_true_label = np.argmax(Y_true, axis=-1)
    Y_pred_label = np.argmax(Y_pred, axis=-1)
    
    # global metrics
    TP = metrics.accuracy_score(Y_true_label, Y_pred_label, normalize=False)
    accuracy = metrics.accuracy_score(Y_true_label, Y_pred_label)
    micro_Fscore = metrics.f1_score(Y_true_label, Y_pred_label, average='micro')
    macro_Fscore = metrics.f1_score(Y_true_label, Y_pred_label, average='macro')
    weighted_Fscore = metrics.f1_score(Y_true_label, Y_pred_label, average='weighted')

    # compile metrics in dict
    metrics_ = dict(error=error, TP=TP, accuracy=accuracy, micro_Fscore=micro_Fscore, macro_Fscore=macro_Fscore, weighted_Fscore=weighted_Fscore)
    
    # confusion matrix for each class
    multiclass_cm = metrics.multilabel_confusion_matrix(Y_true_label, Y_pred_label, labels=list(class_idx.values()))
    multiclass_cm = metrics.multilabel_confusion_matrix(Y_true_label, Y_pred_label)

    # computes binary metrics for each class (one versus all)
    for k, i in class_idx.items():
        
        # statistics from sklearn confusion matrix
        tn, fp, fn, tp = multiclass_cm[i].ravel()

        # metrics
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fnr = fn / (fn + tp)
        fscore = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        
        metrics_.update({
            "{}_precision".format(k): precision,
            "{}_recall".format(k): recall,
            "{}_fscore".format(k): fscore,
            "{}_fnr".format(k): fnr,
            })

    return metrics_

def parse_args():

    parser = argparse.ArgumentParser('Argument for BreastPathQ: Supervised Fine-Tuning/Evaluation')

    parser.add_argument('--data', type=str, required=True, help='path to the data')
    parser.add_argument('--wsi', type=str, required=True, help='path to the WSIs folder')
    parser.add_argument('--output', type=str, required=True, help='path to the file to save results')
    parser.add_argument('--gpu', type=str, default='', help='GPUs to use (e.g. 0)')

    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use.')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training.')

    # model definition
    parser.add_argument('--model', type=str, default='resnet18', help='choice of network architecture.')
    parser.add_argument('--mode', type=str, default='fine-tuning', help='fine-tuning/evaluation')
    parser.add_argument('--num_classes', type=int, required=True, help='# of classes.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size.')
    parser.add_argument('--model_path', type=str, required=True, help='path to load model')    
    parser.add_argument('--image_size', default=256, type=int, help='patch size width 256')
    parser.add_argument('--labels', type=str, required=True, help='path to the label CSV files')
    args = parser.parse_args()

    return args


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labels = pd.read_csv(args.labels).set_index('patient_id')['label'].to_dict()

    model = net.TripletNet_Finetune(args.model)
    classifier = net.FinetuneResNet(args.num_classes)

    model = torch.nn.DataParallel(model)
    classifier = torch.nn.DataParallel(classifier)

    # Load fine-tuned model
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state['model_student'])
    classifier.load_state_dict(state['classifier_student'])

    model.eval()
    classifier.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

    # Test set
    test_transforms = transforms.Compose([
        transforms.Resize(size=args.image_size),
        transforms.ToTensor(),
        ])

    result_per_slide = {}
    list_of_slides = os.listdir(args.data)
    for i, slide in enumerate(list_of_slides):

        print('slide {} ({}/{})'.format(slide, i+1, len(list_of_slides)))
        
        patient = re.findall('\d+', slide)[0]

        data = pd.read_csv(os.path.join(args.data, slide))
        wsi = openslide.OpenSlide(os.path.join(args.wsi, os.path.splitext(slide)[0] + '.tif'))

        print('load patches..')
        patches = []
        for i, p in data.iterrows():
            coord = (p['x'], p['y'])
            coord = scale_coordinates(wsi, coord, p['level'], 0)
            img = wsi.read_region(coord, p['level'], (p['size'],p['size'])).convert('RGB')
            patches.append(img)

        patches = map(test_transforms, patches)
        patches = torch.stack(list(patches), dim=0)
        
        print('classify patches..')
        y_pred = []
        for batch in torch.split(patches, args.batch_size):
            input = batch.to(device=device, dtype=torch.float)
            with torch.no_grad():
                feats = model(input)
                output = classifier(feats)
                y_pred.append(output.to(device='cpu').numpy())
            
        y_pred = np.concatenate(y_pred, axis=0)
        y_pred = np.argmax(y_pred, axis=-1)   # convert samples probabilities to class index
        y_pred = np.bincount(y_pred, minlength=len(slide_class_idx)).astype('float32')   # count number of predicted samples per class
        y_pred /= y_pred.sum()   # normalize to convert to probabilities

        true_idx = slide_class_idx[labels[int(patient)]]
        y_true = np.zeros(len(slide_class_idx), dtype='int32')
        y_true[true_idx] = 1

        result_per_slide[patient] = (y_true, y_pred)

    # calculates metrics on patients
    y_true, y_pred = zip(*result_per_slide.values())
    y_true = np.stack(y_true)
    y_pred = np.stack(y_pred)
    patients_metrics = metrics_fn(y_true, y_pred, slide_class_idx)

    # add model result
    df = pd.DataFrame([patients_metrics])

    # round floats to 2 decimals
    df = df.round(decimals=2)

    # save results in a CSV file
    df.to_csv(args.output, sep=';')

if __name__ == "__main__":

    args = parse_args()

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
