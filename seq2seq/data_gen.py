len = 50
with open('/home/ouyangzhihao/sss/AAAI/common/Mag_Data/Music_Text_Datasets/Bach/Bach_train.txt','r') as fin, open('reverse_train_data','w') as fout:
    Music = fin.readline()
    for i in range(10000):
        melody = Music[i*len:(i+1)*len]
        fout.write(melody+'\n')