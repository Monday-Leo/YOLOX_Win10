import os
import random

train_percent = 4

xmlfilepath = './VOCdevkit/VOC2021/Annotations'
txtsavepath = './VOCdevkit/VOC2021/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)
test_dir = './VOCdevkit/VOC2021/ImageSets/Main/test.txt'
train_dir = './VOCdevkit/VOC2021/ImageSets/Main/trainval.txt'
ftest = open(test_dir, 'w')
ftrain = open(train_dir, 'w')

def ReadFileDatas(dir):
    FileNamelist = []
    file = open(dir,'r+')
    for line in file:
        line=line.strip('\n') #删除每一行的\n
        FileNamelist.append(line)
    #print('len ( FileNamelist ) = ' ,len(FileNamelist))
    file.close()
    return FileNamelist
 
def WriteDatasToFile(listInfo,dir):
    file_handle=open(dir,mode='w')
    for idx in range(len(listInfo)):
        str = listInfo[idx]
        #查找最后一个 “_”的位置
        ndex = str.rfind('_')
        #print('ndex = ',ndex)
        #截取字符串
        str_houZhui = str[(ndex+1):]
        #print('str_houZhui = ',str_houZhui)
        str_Result = str  + '\n'           #+ str_houZhui+'\n'
        #print(str_Result)
        file_handle.write(str_Result)
    file_handle.close()

for i in range(len(total_xml)):
    name = total_xml[i][:-4]
    if i % (train_percent+1) != 0:
        # ftrain.write("./JPEGImages/"+name+".jpg ./Annotations/"+name+".xml\n")
        ftrain.write(name + "\n")
    else:
        # ftest.write("./JPEGImages/"+name+".jpg ./Annotations/"+name+".xml\n")
        ftest.write(name + "\n")
ftrain.close()
ftest.close()

listFileInfo = ReadFileDatas(train_dir)
random.shuffle(listFileInfo)
WriteDatasToFile(listFileInfo,train_dir)
listFileInfo = ReadFileDatas(test_dir)
random.shuffle(listFileInfo)
WriteDatasToFile(listFileInfo,test_dir)