import matplotlib.pyplot as plt
import cv2
import sys
import os
from math import ceil,floor
import numpy as np
from random import choice
import pickle




def main(img, rows, cols, angle, scaling):

  # ROTATE
  M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
  new = cv2.warpAffine(img,M,(cols,rows))

  # RESCALE by s
  # shrinking uses  cv2.INTER_AREA 
  # zooming uses cv2.INTER_CUBIC (slow) OR cv2.INTER_LINEAR
  #
  slow = False
  if slow: zoom_inter = cv2.INTER_CUBIC
  else: zoom_inter = cv2.INTER_LINEAR
  if scaling<1:
    new = cv2.resize(new,(0,0),fx=scaling, fy=scaling, interpolation = cv2.INTER_AREA)
    temp_rows, temp_cols = new.shape
    if (rows-temp_rows)%2==0:
      top,bottom = [(rows-temp_rows)//2]*2
    else:
      top,bottom = floor((rows-temp_rows)/2), ceil((rows-temp_rows)/2) 
    if (cols-temp_cols)%2==0:
      left,right = [(cols-temp_cols)//2]*2
    else:
      left,right = floor((cols-temp_cols)/2), ceil((cols-temp_cols)/2) 
    new = cv2.copyMakeBorder(new,top, bottom, left, right,  cv2.BORDER_CONSTANT, value=0)
  elif scaling>1:
    new = cv2.resize(new,(0,0),fx=scaling, fy=scaling, interpolation = zoom_inter)
    temp_rows, temp_cols = new.shape
    new = new[(temp_rows-rows)//2:(temp_rows+rows)//2, (temp_cols-cols)//2:(temp_cols+cols)//2]
    new = cv2.resize(new,(rows,cols), interpolation = cv2.INTER_AREA)

  return new

def viewer(img, new):
  f, ax = plt.subplots(1,2)
  ax[0].imshow(img, cmap='gray')
  ax[1].imshow(new, cmap='gray')
  plt.show()

if __name__=='__main__':
  os.chdir('images')
  images = {name:cv2.imread(name,0) for name in os.listdir() if name[-3:] in ['png','jpg']}
  images_names = list(images.keys())
  os.chdir('..')
  print(images_names)
  data = {'angle':[], 'scaling':[],
          'name':[],  'image':[],'source':[]}
  try: 
    view = sys.argv[1]
  except IndexError: pass

  #angles = [-90,-45,0,45,90]
  #scalings = [0.7,0.85,1,1.15,1.3]
  angles = np.linspace(-90,90,180)
  scalings = [round(2**x,3) for x in np.linspace(-1,1,50)]
  for x in range(20000):
    img_name = choice(images_names)
    img = images[img_name]
    rows,cols = img.shape
    angle = choice(angles)
    scaling = choice(scalings)
    result = main(img, rows, cols, angle, scaling)
    data['angle'].append(angle)
    data['source'].append(img_name)
    data['scaling'].append(scaling)
    data['image'].append(result)
    name_index=0
    name = f'{angle}_{scaling}_{name_index}'
    while name in data['name']:
      name_index+=1
      name = f'{angle}_{scaling}_{name_index}'
    data['name'].append(name)
    try:
      if view: viewer(img, result)  
    except NameError: pass
  with open('database.pkl','wb') as f:
    pickle.dump(data,f)
    
