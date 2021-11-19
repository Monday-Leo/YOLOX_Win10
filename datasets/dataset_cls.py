import os
import random

trainval_percent = 0.15
train_percent = 0.85
xmlfilepath = './VOCdevkit/VOC2021/Annotations'
txtsavepath = './VOCdevkit/VOC2021/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftest = open('./VOCdevkit/VOC2021/ImageSets/Main/test.txt', 'w')
ftrain = open('./VOCdevkit/VOC2021/ImageSets/Main/trainval.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftest.write(name)
    else:
        ftrain.write(name)

ftrain.close()
ftest.close()


