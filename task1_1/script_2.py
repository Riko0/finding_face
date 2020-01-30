from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import os.path as osp
import torch
import tqdm
import pandas as pd
import csv
import argparse
from zipfile import ZipFile
import shutil

#скрипт без обучения, можно дообучить, тогда качество будет выше

#parsing input args
input_zip = ""
parser = argparse.ArgumentParser(description='')
parser.add_argument('archive_name', action='store', nargs='?',
    help='name of zip with images', default="train.zip")

args = parser.parse_args()
input_zip = args.archive_name

#extraicting zip to temp
temp_folder = 'temp'
with ZipFile(input_zip, 'r') as zipObj:
   # Extract all the contents of zip file in different directory
   zipObj.extractall(temp_folder)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

#folder with faces to look for
input_folder = "input"
referent_images = [osp.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('jpg')]

for oleg_image in referent_images:
    olegs = []
    oleg = Image.open(oleg_image)
    oleg_faces = mtcnn(oleg)  
    for face in oleg_faces:
        olegs.append(face)    
    olegs = torch.stack(olegs).to(device)
    oleg_embeddings = resnet(olegs).detach().cpu()

#images 
images = []
for path, dirs, files in os.walk(temp_folder):
    for file in files:
        if file.endswith('jpg'):
            images.append(osp.join(path, file))

result = []
threshold = 1.1

for image in tqdm.tqdm(images):
    img = Image.open(image)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    
    faces_on_frame = []
    
    faces = mtcnn(img)
    for face in faces:
        faces_on_frame.append(face)
        
    faces_on_frame = torch.stack(faces_on_frame).to(device)
    embeddings = resnet(faces_on_frame).detach().cpu()
    
    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in oleg_embeddings]

    if min(min(l) for l in dists) < 1.1:
        result.append([osp.split(image)[-1], 1])
    else:
        result.append([osp.split(image)[-1], 0])

result_file = open("output.csv",'w', newline='')
wr = csv.writer(result_file)
for item in result:
    wr.writerow(item)
 
shutil.rmtree(temp_folder)

print("Done! Results are in output.csv")