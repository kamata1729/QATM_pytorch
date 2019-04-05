# Pytorch Implementation of QATM:Quality-Aware Template Matching For Deep Learning

arxiv: https://arxiv.org/abs/1903.07254  
original code (tensorflow+keras): https://github.com/cplusx/QATM  
Qiita(Japanese): https://qiita.com/kamata1729/items/11fd55992c740526f6fc

# Dependencies

* torch(1.0.0)
* torchvision(0.2.1)
* cv2
* seaborn
* scipy
* sklearn

# Usage

see [`qatm_pytorch.ipynb`](https://github.com/kamata1729/QATM_pytorch/blob/master/qatm_pytorch.ipynb)

or


```
python qatm.py -s sample/sample1.jpg -t template/template1_1.png template/template1_2.png template/template1_dummy.png --cuda
```

* Add `--cuda` option to use GPU
* Add `-s`/`--sample_image` to specify sample image  
    **only single** sample image can be specified in this present implementation  
* Add `-t`/`--template_images` to specify template image(s)  
    **single or multi** images can be specified as template image(s), but sizes of template images must be same  
  
**[notice]** If neither `-s` nor `-t` is specified, demo program will be executed, which is the same as:
```
python qatm.py -s sample/sample1.jpg -t template/template1_1.png template/template1_2.png template/template1_dummy.png
```

* `--thresh` and `--alpha` option can also be added

# Result of Demo
`template1_1.png` and `template1_2.png` are contained in `sample1.jpg`, however, `template1_dummy.png` is a dummy and not contained

|template1_1.png|template1_2.png|template1_dummy.png|
|---|---|---|
|![](https://i.imgur.com/lP0Wy4I.png)|![](https://i.imgur.com/xDJoQOz.png)|![](https://i.imgur.com/p10g33j.png)|

![](https://i.imgur.com/7Ln2z0H.png)
