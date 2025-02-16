import requests
import pandas as pd
import time
import cv2
import os
import random

images = 20
dataset = pd.read_csv("data/painting_dataset_2021.csv")

def download(name,url):
    res = requests.get(url)
    with open('data/raw/'+name+'.jpg', 'wb') as f:
        f.write(res._content)
        return f
    
def clean():
    for f in os.listdir('data/raw'):
        im = cv2.imread('data/raw/'+f)
        im = cv2.resize(im,(256,256))
        cv2.imwrite('data/clean/'+f,im)

ids = list(range(0,len(dataset)-1))
random.shuffle(ids)

for i in ids[:images]:
    download(str(i),dataset.iloc[i]["Image URL"]) 
    time.sleep(1)

clean()