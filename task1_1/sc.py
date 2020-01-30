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
from PIL import Image, ImageDraw
import face_recognition
import numpy

with open('train_labels.csv', 'r') as f:
    reader = csv.reader(f)
    file_list = list(reader)
    
right_files = []

for file in file_list:
    if file[1] == '1':
        right_files.append(file[0])

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

#folder with faces to look for
input_folder = "input"
referent_images = [osp.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('jpg')]

oleg_embeddings = []

for oleg_image in referent_images:
    # Load some sample pictures and learn how to recognize them.
    oleg_im = face_recognition.load_image_file(oleg_image)
    oleg_embeddings.append(face_recognition.face_encodings(oleg_im)[0])

#images 
images = []
for path, dirs, files in os.walk(temp_folder):
    for file in files:
        if file.endswith('jpg'):
            images.append(osp.join(path, file))

result = []

pics_with_oleg = []

for image in tqdm.tqdm(images):
    img = Image.open(image)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    
    open_cv_image = numpy.array(img) 
    face_locations = face_recognition.face_locations(open_cv_image)
    face_encodings = face_recognition.face_encodings(open_cv_image, face_locations)
    
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(oleg_embeddings, face_encoding, tolerance=0.50)
        
        flag = False
        for m in match:
            if m:
                flag = True
                break
                
    if flag:
        result.append([osp.split(image)[-1], 1])
        pics_with_oleg.append(osp.split(image)[-1])
    else:
        result.append([osp.split(image)[-1], 0])

result_file = open("output.csv",'w', newline='')
wr = csv.writer(result_file)
for item in result:
    wr.writerow(item)
 
shutil.rmtree(temp_folder)

def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 
    
extra = Diff(pics_with_oleg, right_files)

print("extra", len(extra), extra)

needed = Diff(right_files, pics_with_oleg)

print("needed", len(needed), needed)

print("Done! Results are in output.csv")