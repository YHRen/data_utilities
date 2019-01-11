"""
- Split the xView data into training and validation (default ratio 9:1)
- Convert xView geojson annotation to yolov3 format.
- Divide the large TIF images into smaller images.

Author: Yihui (Ray) Ren
Email : yren@bnl.gov
NOTE  : not tested yet...
"""

import json
from PIL import Image
from pathlib import Path
import numpy as np
import wv_util as wv
#import vis
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse

def convert_to_yolo_box(sz, b):
    ans = b.copy()
    ans[:,0] = (b[:,2] + b[:,0])/2.0 - 1
    ans[:,1] = (b[:,3] + b[:,1])/2.0 - 1
    ans[:,2] = b[:,2] - b[:,0]
    ans[:,3] = b[:,3] - b[:,1]
    ans[:,(0,2)]  /= sz[0]
    ans[:,(1,3)]  /= sz[1]
    return ans

def vis_image(idx):
    global img_dir, img_files
    im = np.array(Image.open(img_dir/img_files[idx]))
    bbox = coords[img_fn==img_files[idx]]
    cls  = classes[img_fn==img_files[idx]].astype(np.int)
    print("file name:",img_files[idx],"file size:", im.shape, "num bbox:", len(bbox))
    print('cls-id, cls-name, occur.')
    print( '\n'.join([', '.join(map(str,(int(x),cid2name[x], y))) for x,y in Counter(cls).most_common()]))
    vis.show_img_anno(im,zip(cls, bbox), figsize=(im.shape[0]/200,im.shape[1]/200) )

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--geojson', help='the xview_train.geojson file', type=str)
parser.add_argument('-m', '--images', help='the xview train image folder', type=str)
parser.add_argument('-d', '--id2name', help='the id2name txt file', type=str)
parser.add_argument('-o', '--output', help='output directory', type=str, default='./')
parser.add_argument('-r', '--resolution', help='output image resolution', type=int, default=672)
args = parser.parse_args()

## load all annotations
coords, img_fn, classes = wv.get_labels(args.geojson)
img_files = list(set(img_fn))
print("total image files: ", len(img_files), "first 5 image files = ",img_files[:5])
cls_cnt = Counter(classes).most_common()
## Load id2name
cid2name = {}
with open(args.id2name,'r') as f:
    for line in f:
        x, y= line.strip().split(':')
        cid2name[int(x)] = y
# map id to consecutive ids
idremap = {x:y for x,y in zip(sorted(cid2name.keys()), range(len(cid2name)))}
img_dir = args.images

## divide the TIF images into smaller jpg images
if True:
    tif_path = Path(args.images)
    fname2idx = { x:i for i,x in enumerate(tif_path.iterdir())}
    idx2fname = { fname2idx[x]:x for x in fname2idx.keys()}
    bmtx = np.zeros((len(fname2idx), len(idremap)))
    for fname in tqdm(tif_path.iterdir()):
        i = fname2idx[fname]
        cls = set(classes[img_fn == fname.name])
        cls = [idremap[x] for x in cls]
        bmtx[i, cls]= True

    sd = 179 ## random seed to split
    trn_sz = 761 ## 90%
    np.random.seed(sd)
    idx_np = np.asarray(range(bmtx.shape[0]))
    np.random.shuffle(idx_np)
    flag1 = bmtx[idx_np[:trn_sz],:].any(axis=0).all()
    flag2 = bmtx[idx_np[trn_sz:],:].any(axis=0).all()
    print(sd, flag1, flag2)


    trn_path = Path(args.output)/'chip_train/'
    val_path = Path(args.output)/'chip_valid/'
    trn_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    resolution = [args.resolution] #416 # 672

    ## Train
    out_path = trn_path
    for idx in tqdm(idx_np[:trn_sz]):
        fname = idx2fname[idx]
        arr = wv.get_image(fname)
        for res in resolution:
            im, box, classes_final = wv.chip_image(arr, coords[img_fn==fname.name], classes[img_fn==fname.name], (res,res))
            for idx, (x, b, c) in enumerate(zip(im,box.values(), classes_final.values())):
                if len(c) == 1 and c[0] == 0: continue ## no instance
                else:
                    im_data = Image.fromarray(x)
                    tmp_fn = '_'.join(map(str,[fname.stem, idx, res]))
                    im_data.save(out_path/(tmp_fn+'.jpg'))
                    with open( str(out_path/(tmp_fn+'.txt')), 'w') as anno_f:
                        yolob = convert_to_yolo_box( x.shape[:2], b)
                        for i in range(len(c)):
                            anno_f.write( str(idremap[int(c[i])])+ ' '+ ' '.join(map(str, yolob[i])) + '\n' )

    ## Valid
    out_path = val_path
    for idx in tqdm(idx_np[trn_sz:]):
        fname = idx2fname[idx]
        arr = wv.get_image(fname)
        for res in resolution:
            im, box, classes_final = wv.chip_image(arr, coords[img_fn==fname.name], classes[img_fn==fname.name], (res,res))
            for idx, (x, b, c) in enumerate(zip(im,box.values(), classes_final.values())):
                if len(c) == 1 and c[0] == 0: continue ## no instance
                else:
                    im_data = Image.fromarray(x)
                    tmp_fn = '_'.join(map(str,[fname.stem, idx, res]))
                    im_data.save(out_path/(tmp_fn+'.jpg'))
                    with open( str(out_path/(tmp_fn+'.txt')), 'w') as anno_f:
                        yolob = convert_to_yolo_box( x.shape[:2], b)
                        for i in range(len(c)):
                            anno_f.write( str(idremap[int(c[i])])+ ' '+ ' '.join(map(str, yolob[i])) + '\n' )
