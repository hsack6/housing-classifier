# -*- coding: utf-8 -*-
import pathlib
import numpy as np
from keras.utils import to_categorical
import random
from keras.preprocessing.image import img_to_array, load_img

def get_path(path,list=[]):
    # 再帰的にiterdirを呼び出し、全てのファイルを取得
    # ファイルなら、ファイルの絶対パスをリストに追加
    # path以下の全てのファイルの絶対パスのリストを返す
    if path.is_file():
        list.append(str(path.resolve()))
    # ディレクトリなら、iterdirでこの関数を呼び出す
    elif path.is_dir():
        for p in path.iterdir():
            get_path(p,list)
    return list

class NpyDataGenerator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = []
        self.labels = []

    def flow_from_directory(self, directory, classes, batch_size=32):
        # LabelEncode(classをint型に変換)するためのdict
        classes = {v: i for i, v in enumerate(classes)}
        pathList = get_path(directory,[])

        while True:
            # ディレクトリからラベル、名前、データのパスを取り出す
            random.shuffle(pathList)
            for path in pathList:
                label_data_path = path.replace(str(directory)+"/", "")
                label, name = label_data_path.split('/')
                #npydata = np.load(str(directory)+"/"+label_data_path)
                npydata = img_to_array(load_img(str(directory)+"/"+label_data_path, target_size=(224,224)))
                #print(name+":"+label+":",npydata.shape)

                self.data.append(npydata)
                self.labels.append(to_categorical(classes[label], len(classes)))

                # ここまでを繰り返し行い、batch_sizeの数だけ配列(self.data, self.labels)に格納
                # batch_sizeの数だけ格納されたら、戻り値として返し、配列(self.data, self.labels)を空にする
                if len(self.data) == batch_size:
                    inputs = np.asarray(self.data, dtype=np.float32)
                    targets = np.asarray(self.labels, dtype=np.float32)
                    targets = np.reshape(targets, (batch_size, -1))
                    self.reset()
                    yield inputs, targets
