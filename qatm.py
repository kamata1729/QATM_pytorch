from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
import argparse
from utils import *
from glob import glob
import gc
import os

# +
# import functions and classes from qatm_pytorch.py
print("import qatm_pytorch.py...")
import ast
import types
import sys

with open("qatm_pytorch.py") as f:
       p = ast.parse(f.read())

for node in p.body[:]:
    if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        p.body.remove(node)

module = types.ModuleType("mod")
code = compile(p, "mod.py", 'exec')
sys.modules["mod"] = module
exec(code,  module.__dict__)

from mod import *
# -

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-s', '--sample_image', default='sample/sample1.jpg')
    parser.add_argument('-t', '--template_images_dir', default='template/')
    parser.add_argument('-ss', '--sample_images_dir')
    parser.add_argument('-r', '--result_images_dir', default='result/')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh_csv', type=str, default='thresh_template.csv')
    args = parser.parse_args()
    
    template_dir = args.template_images_dir
    result_path = args.result_images_dir
    if not os.path.isdir(result_path):
        os.mkdir(result_path)    

    print("define model...")
    model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=args.cuda)
    
    if not args.sample_images_dir:
        print('One Sample Image Is Inputted')
        image_path = args.sample_image
        dataset = ImageDataset(Path(template_dir), image_path, thresh_csv='thresh_template.csv')
        print("calculate score...")
        scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
        print("nms...")
        boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
        _ = plot_result_multi(dataset.image_raw, boxes, indices, show=False, save_name='result.png')
        print("result.png was saved")

    else:
        print('Image Directory Is Inputted')
        sample_images_dir = args.sample_images_dir
        images = glob(os.path.join(sample_images_dir,'*'))
        i=1
        for image in images:
            print('-----',i,'/',len(images),'-----')
            image_name = image.split('/')[-1].split('.')[0]
            print('Sample Image:',image_name,'Processing...')
            dataset = ImageDataset(Path(template_dir), image)
            print("calculate score...")
            scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
            print("nms...")
            boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
            d_img = plot_result_multi(dataset.image_raw, boxes, indices, show=True, save_name=os.path.join(result_path,image_name)+'.png')
            print("result image was saved")
            del(dataset)
            del(d_img)
            gc.collect()
            torch.cuda.empty_cache()
            i+=1