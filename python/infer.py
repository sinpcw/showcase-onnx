import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
import onnxruntime as ort
from tqdm.auto import tqdm

class CFG:
    image_size = 224
    image_dir = 'data/dog-breed-identification/train'

class Inference:
    def __init__(self, onnx_path):
        # CPUのみの場合:
        # self.session = ort.InferenceSession('model.onnx', providers=[ 'CPUExecutionProvider' ])
        # GPUを利用する場合:
        self.session = ort.InferenceSession('model.onnx', providers=[ 'CUDAExecutionProvider', 'CPUExecutionProvider' ])

    def __call__(self, x):
        ret = self.session.run([ 'output' ], { 'input1' : x })
        return ret[0]

class Preprocess:
    def __init__(self):
        self.ops = A.Compose([
            A.Resize(CFG.image_size, CFG.image_size, p=1.00),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
        ])

    def __call__(self, image):
        img = self.ops(image=image)['image'].astype(np.float32)
        img = img.transpose(2, 0, 1)[None, :, :, :]
        return img

def loadimg(file):
    img = cv2.imread(file)
    out = cv2.resize(img, (224, 224))
    cv2.imwrite(os.path.join(CFG.trans_dir, os.path.basename(file)), out)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def main():
    # 推論対象ファイル (ここではvalidation.csvのファイルを対象とする)
    valid_df = pd.read_csv('sample_valid.csv')
    files = [ os.path.join(CFG.image_dir, f + '.jpg') for f in valid_df.iloc[:, 0].values ]
    # onnxファイルからモデルを取得
    infer = Inference('model.onnx')
    # preprocess:
    preprocess = Preprocess()
    result_ids = [ ]
    result_cls = [ ]
    for _, file in enumerate(tqdm(files)):
        x = loadimg(file)
        x = preprocess(x)
        y = infer(x)
        # (1, 120) -> (1, )
        y = np.argmax(y, axis=1)
        y = y[0]
        result_ids.append(os.path.basename(file)[:-4])
        result_cls.append(y)
    # DataFrameの作成:
    result_df = pd.DataFrame({
        'id' : result_ids,
        'predict_id' : result_cls,
    })
    # valid_dfとマージして結果を集約 / 列順をソート
    result_df = pd.merge(result_df, valid_df, how='left', on='id')
    result_df = result_df.loc[:, [ 'id', 'breed', 'class_id', 'predict_id' ]]
    result_df.to_csv('result.csv', index=False)

""" エントリポイント """
if __name__ == "__main__":
    main()