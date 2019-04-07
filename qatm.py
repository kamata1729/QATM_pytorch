import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, utils
import copy
import argparse

from utils import *
from lib import *


class ImageDataset(torch.utils.data.Dataset):
    """
    define Dataset
    """
    def __init__(self, template_names: list, image_name: str, transform=None):
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
        self.template_names = template_names
        self.image_name = image_name
        
        self.image_raw = cv2.imread(self.image_name)
        if self.transform:
            self.image = self.transform(self.image_raw)
        
    def __len__(self):
        return len(self.template_names)
    
    def __getitem__(self, idx):
        template = cv2.imread(self.template_names[idx])
        if self.transform:
            template = self.transform(template)
        return {'image': self.image.unsqueeze(0), 
                    'image_raw': self.image_raw, 
                    'template': template.unsqueeze(0), 
                    'template_name': self.template_names[idx], 
                    'template_h': template.size()[-2],
                   'template_w': template.size()[-1]}


class Featex():
    """
    extract template/image feature through pretrained model
    """
    def __init__(self, model, use_cuda):
        self.use_cuda = use_cuda
        self.feature1 = None
        self.feature2 = None
        self.model= copy.deepcopy(model.eval())
        self.model = self.model[:17]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model[2].register_forward_hook(self.save_feature1)
        self.model[16].register_forward_hook(self.save_feature2)
        
    def save_feature1(self, module, input, output):
        self.feature1 = output.detach()
    
    def save_feature2(self, module, input, output):
        self.feature2 = output.detach()
        
    def __call__(self, input, mode='big'):
        if self.use_cuda:
            input = input.cuda()
        _ = self.model(input)
        if mode=='big':
            # resize feature1 to the same size of feature2
            self.feature1 = F.interpolate(self.feature1, size=(self.feature2.size()[2], self.feature2.size()[3]), mode='bilinear', align_corners=True)
        else:        
            # resize feature2 to the same size of feature1
            self.feature2 = F.interpolate(self.feature2, size=(self.feature1.size()[2], self.feature1.size()[3]), mode='bilinear', align_corners=True)
        return torch.cat((self.feature1, self.feature2), dim=1)
    
    
class MyNormLayer():
    """
    normalize feature maps
    """
    def __call__(self, x1, x2):
        bs, _ , H, W = x1.size()
        _, _, h, w = x2.size()
        x1 = x1.view(bs, -1, H*W)
        x2 = x2.view(bs, -1, h*w)
        concat = torch.cat((x1, x2), dim=2)
        x_mean = torch.mean(concat, dim=2, keepdim=True)
        x_std = torch.std(concat, dim=2, keepdim=True)
        x1 = (x1 - x_mean) / x_std
        x2 = (x2 - x_mean) / x_std
        x1 = x1.view(bs, -1, H, W)
        x2 = x2.view(bs, -1, h, w)
        return [x1, x2]
    
    
class QATM():
    """
    QATM module to get configuration map
    """
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, x):
        batch_size, ref_row, ref_col, qry_row, qry_col = x.size()
        x = x.view(batch_size, ref_row*ref_col, qry_row*qry_col)
        xm_ref = x - torch.max(x, dim=1, keepdim=True)[0]
        xm_qry = x - torch.max(x, dim=2, keepdim=True)[0]
        confidence = torch.sqrt(F.softmax(self.alpha*xm_ref, dim=1) * F.softmax(self.alpha * xm_qry, dim=2))
        conf_values, ind3 = torch.topk(confidence, 1)
        ind1, ind2 = torch.meshgrid(torch.arange(batch_size), torch.arange(ref_row*ref_col))
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
        ind3 = ind3.flatten()
        if x.is_cuda:
            ind1 = ind1.cuda()
            ind2 = ind2.cuda()
        
        values = confidence[ind1, ind2, ind3]
        values = torch.reshape(values, [batch_size, ref_row, ref_col, 1])
        return values
    def compute_output_shape( self, input_shape ):
        bs, H, W, _, _ = input_shape
        return (bs, H, W, 1)
    
    
class CreateModel():
    """
    create model and return configuration maps
    """
    def __init__(self, alpha, model, use_cuda):
        self.alpha = alpha
        self.featex = Featex(model, use_cuda)
    def __call__(self, template, image):
        T_feat = self.featex(template)
        I_feat = self.featex(image)
        conf_maps = None
        batchsize_T = T_feat.size()[0]
        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            I_feat, T_feat_i = MyNormLayer()(I_feat, T_feat_i)
            dist = torch.einsum("xcab,xcde->xabde", I_feat / torch.norm(I_feat, dim=1, keepdim=True), T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True))
            conf_map = QATM(self.alpha)(dist)
            if conf_maps is None:
                conf_maps = conf_map
            else:
                conf_maps = torch.cat([conf_maps, conf_map], dim=0)
        return conf_maps
    

def run_one_sample(template, image):
    val = model(template, image)
    if val.is_cuda:
        val = val.cpu()
    val = val.numpy()
    val = np.log(val)
    
    batch_size = val.shape[0]
    scores = []
    for i in range(batch_size):
        # compute geometry average on score map
        gray = val[i,:,:,0]
        gray = cv2.resize( gray, (image.size()[-1], image.size()[-2]) )
        h = template.size()[-2]
        w = template.size()[-1]
        score = compute_score( gray, w, h) 
        score[score>-1e-7] = score.min()
        score = np.exp(score / (h*w)) # reverse number range back after computing geometry average
        scores.append(score)
    return np.array(scores)
    
    
def run_multi_sample(dataset):
    scores = None
    w_array = []
    h_array = []
    for data in dataset:
        score = run_one_sample(data['template'], data['image'])
        if scores is None:
            scores = score
        else:
            scores = np.concatenate([scores, score], axis=0)
        w_array.append(data['template_w'])
        h_array.append(data['template_h'])
    return np.array(scores), np.array(w_array), np.array(h_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-s', '--sample_image')
    parser.add_argument('-t', '--template_images', nargs='*')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh', type=float, default=0.8)
    args = parser.parse_args()
    
    image_path = args.sample_image
    # template images must have same size
    template_list = args.template_images
    
    if not image_path or not template_list:
        print("Either --sample_image or --template_images is not specified, so demo program is running...")
        image_path = 'sample/sample1.jpg'
        template_list = ['template/template1_1.png', 
                 'template/template1_2.png', 
                 'template/template1_3.png', 
                 'template/template1_4.png',
                 'template/template1_dummy.png']
        
    dataset = ImageDataset(template_list, image_path)
    
    print("define model...")
    model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=args.cuda)
    print("calculate score...")
    scores, w_array, h_array = run_multi_sample(dataset)
    print("nms...")
    boxes, indices = nms_multi(scores, w_array, h_array, thresh=args.thresh)
    _ = plot_result_multi(dataset.image_raw, boxes, indices, show=False, save_name='result.png')
    print("result.png was saved")