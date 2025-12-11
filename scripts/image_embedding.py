import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import gc
device = "cuda" if torch.cuda.is_available() else "cpu"

test = pd.read_csv('../data/test.csv')
test['activation_date'] = pd.to_datetime(test['activation_date'])
test["day"] = test['activation_date'].dt.day
test["month"] = test["activation_date"].dt.month
test['month'] = test['activation_date'].dt.weekday
test['year'] = test["activation_date"].dt.year
test["dayofyear"] = test['activation_date'].dt.dayofyear 

from transformers import ViTImageProcessor, ViTModel

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)


from PIL import Image
from tqdm import tqdm
import pickle
res = []
for index, row in tqdm(test.iterrows(), total=len(test)):
    if row['image'] is None:
        res.append(None)
        continue
    try:
        image = Image.open('../test_jpg/' + row['image'] + '.jpg')
    except:
        res.append(None)
        if len(res) % 200 == 0:
            with open("../data/vit_test_jpg_" + str(index), "wb") as fp:
                pickle.dump(res, fp)
            del res
            gc.collect()
            res = []
        continue
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    res.append(outputs.last_hidden_state)
    del inputs
    del outputs
    gc.collect()
    torch.cuda.empty_cache()
    if len(res) % 200 == 0:
        with open("../data/vit_test_jpg_" + str(index), "wb") as fp:
            pickle.dump(res, fp)
        del res
        gc.collect()
        res = []