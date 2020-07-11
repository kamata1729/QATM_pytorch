# Pytorch Non-Official Implementation of QATM:Quality-Aware Template Matching For Deep Learning

arxiv: https://arxiv.org/abs/1903.07254  
original code (tensorflow+keras): https://github.com/cplusx/QATM  
Qiita(Japanese): https://qiita.com/kamata1729/items/11fd55992c740526f6fc

# Dependencies

* torch(1.0.0)
* torchvision(0.2.1)
* cv2
* seaborn
* sklearn
* pathlib

# Usage

See [`qatm_pytorch.ipynb`](https://github.com/kamata1729/QATM_pytorch/blob/master/qatm_pytorch.ipynb)

or


```
python qatm.py -s sample/sample1.jpg -t template --cuda
```

* Add `--cuda` option to use GPU
* Add `-s`/`--sample_image` to specify sample image  
    **only single** sample image can be specified in this present implementation  
* Add `-t`/`--template_images_dir` to specify template image dir  
  
**[notice]** If neither `-s` nor `-t` is specified, the demo program will be executed, which is the same as:
```
python qatm.py -s sample/sample1.jpg -t template
```

* `--thresh_csv` and `--alpha` option can also be added

# Result of Demo
`template1_1.png` to `template1_4.png` are contained in `sample1.jpg`, however, `template1_dummy.png` is a dummy and not contained

|template1_1.png|template1_2.png|template1_3.png|template1_4.png|template1_dummy.png|
|---|---|---|---|---|
|![](https://i.imgur.com/lP0Wy4I.png)|![](https://i.imgur.com/xDJoQOz.png)|![image.png](https://qiita-image-store.s3.amazonaws.com/0/262908/472c81ae-9afb-db49-a64c-86604cbe0884.png)|![image.png](https://qiita-image-store.s3.amazonaws.com/0/262908/d402a9d2-bbd4-5353-16aa-567b79ca06b8.png)|![](https://i.imgur.com/p10g33j.png)|

![image.png](https://qiita-image-store.s3.amazonaws.com/0/262908/2e4c4b8b-2889-7962-4f35-c313048dc403.png)
