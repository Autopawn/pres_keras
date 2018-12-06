import os
import re

import numpy as np

from PIL import Image

# Get the file names:
DIR = './UTKFace'
fnames = list(os.listdir(DIR))

# Data arrays
x_img = []
x_age = []
x_gender = []
y_race = []

# Regular expression to parse a part of the name
regex = re.compile('(\d+)_(\d+)_(\d+)_.*\.jpg')

# Process each file:
for fname in fnames:
    match = regex.match(fname)
    if match:
        age,gender,race = match.groups()
    else:
        print("Bad match: \"%s\""%fname)
        continue
    x_age.append(int(age))
    x_gender.append(int(gender))
    y_race.append(int(race))
    # Read the image and scale it
    img = Image.open(os.path.join(DIR,fname))
    img = img.convert('RGB').resize((48,48))
    x_img.append(np.array(img,dtype='float')/255.0)

print("Saving arrays...")
np.save('x_img.npy',np.array(x_img,dtype='float'))
np.save('x_age.npy',np.array(x_age,dtype='int'))
np.save('x_gender.npy',np.array(x_gender,dtype='int'))
np.save('y_race.npy',np.array(y_race,dtype='int'))
