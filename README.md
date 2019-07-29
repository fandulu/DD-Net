
# DD-Net 
(A Double-feature Double-motion Network)

## 1.About this code
A lightweight network for body/hand action recognition, implemented by keras tensorflow backend.

## 2.How to use this code
### (1) create an anaconda environment by following command
```
conda env create -f=DD-Net_env.yml
```
### (2) go to the folder of JHMDB or SHREC to play with ipython notebooks.
Note: You can download the raw data and use our code to preprocess them, or, directly use our preprocessed data under /data. 

### An alternative choice:
If you do not have enough resource to run this code, please go to use https://colab.research.google.com/drive/19gq3bUigdxIfyMCoWW93YhLEi1KQlBit. We have the preprocessed data under /data, you can download the data and upload them to colab->files, and then run our code on colab.


## 3.Problems this code try to alleviate
<img src="https://github.com/fandulu/DD-Net/blob/master/demo.png" width="500">

## 4.Performance
|No. parameters | SHREC-14 | SHREC-28 |
| :----: | :----: | :----: |
| 1.82 M | 94.6 | 91.9  |
| 0.15 M | 91.8| 90.0|

|No. parameters | JHMDB|
| :----: | :----: | 
| 0.50 M | 78.0 | 
| 0.15 M | 74.7|

Note: if you want to test the speed, please try to run the model.predict() at leat twice and do not take the speed of first run, the model initialization takes extra time.
## 5.Citation
If you find this code is helpful, thanks for citing our work as,
```
@online{1907.09658,
Author = {Fan Yang and Sakriani Sakti and Yang Wu and Satoshi Nakamura},
Title = {Make Skeleton-based Action Recognition Model Smaller, Faster and Better},
Year = {2019},
Eprint = {1907.09658},
Eprinttype = {arXiv},
}
```
![](demo.gif)
*Hey, take a look!*
