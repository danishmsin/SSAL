import torch
import numpy as np
import sys
sys.path.append('/home/megh/projects/domain-adaptation/SSDA_MME/')
from loaders.data_list import Imagelists_VISDA, return_classlist
from model.basenet import *
from model.resnet import *
from torchvision import transforms
from utils.return_dataset import ResizeImage
import time
import os
#from sklearn.decomposition import PCA
from torch.autograd import Variable
# Defining return dataset function here
net = "resnet34"
root = '../data/multi/'
source = "painting"
target = "clipart"
n_class = 126
image_set_file_test = "/home/megh/projects/domain-adaptation/SSDA_MME/data/txt/multi/unlabeled_target_images_sketch_3.txt"
#model_path = "../freezed_models/alexnet_p2r_ours.ckpt.best.pth.tar"
ours = False

def get_dataset(net,root,image_set_file_test):
    if net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                            transform=data_transforms['test'],
                                            test=True)
    class_list = return_classlist(image_set_file_test)
    num_images = len(target_dataset_unl)
    if net == 'alexnet':
        bs = 1
    else:
        bs = 1

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs, num_workers=3,shuffle=False, drop_last=False)
    return target_loader_unl,class_list


target_loader_unl,class_list = get_dataset(net,root,image_set_file_test)

# Deinfining the pytorch networks
if net == 'resnet34':
    G = resnet34()
    inc = 512
elif net == 'resnet50':
    G = resnet50()
    inc = 2048
elif net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')


if ours: 
    if net == 'resnet34':
        F1 = Predictor_deep_2(num_class=n_class,inc=inc,feat_dim = 50)
        print("Using: Predictor_deep_attributes")
    else:
        F1 = Predictor_attributes(num_class=n_class,inc=inc,feat_dim = 50)
        print("Using: Predictor_attributes")

else:
    if net == 'resnet34':
        F1 = Predictor_deep(num_class=n_class,inc=inc)
        print("Using: Predictor_deep")
    else:
        F1 = Predictor(num_class=n_class, inc=inc, temp=0.05)
        print("Using: Predictor")


G.cuda()
F1.cuda()

ckpt = torch.load("../save_model_ssda/model_sketch_step_2000.pth.tar")

G.load_state_dict(ckpt['G'])
F1.load_state_dict(ckpt['F1'])

im_data_t = torch.FloatTensor(1)
im_data_t = im_data_t.cuda()
im_data_t = Variable(im_data_t)
gt_labels_t = torch.LongTensor(1)
gt_labels_t = gt_labels_t.cuda()
gt_labels_t = Variable(gt_labels_t)

def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
            print(batch_idx)
            torch.cuda.empty_cache()
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.0f}%)\n'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size
print(len(target_loader_unl.dataset))
a,b = test(target_loader_unl)
print(b)
