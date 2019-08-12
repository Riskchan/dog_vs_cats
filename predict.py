import os
import sys
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.models import model_from_json

image_size = 150

result_dir = 'results'

# load model weights
#model = load_model('./results/smallcnn.h5')
# load json and create model
json_file = open('results/smallcnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("results/smallcnn.h5")

# load img
if len(sys.argv) != 2:
    print('invalid argment')
    sys.exit()
else:
    files = os.listdir(sys.argv[1])
    for filename in files:
        X = []
        img = Image.open(sys.argv[1] + "/" + filename)
        img = img.convert("RGB")
        img = img.resize((image_size, image_size))
        in_data = np.asarray(img)
        X.append(in_data)
        X = np.array(X)

        predict =  model.predict(X)
        for pre in predict:
            y = pre.argmax()
            print("%s: %d" % (filename, y))
            
    """     
    im_jpg = sys.argv[1]
    X = []
    for file_name in sys.argv[1:]:
        img = Image.open(file_name)
        img = img.convert("RGB")
        img = img.resize((image_size, image_size))
        in_data = np.asarray(img)
        X.append(in_data)
    
    X = np.array(X)

predict =  model.predict(X)
for pre in predict:
    y = pre.argmax()
    print(y)
    """
