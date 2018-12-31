import cv2
from keras.models import load_model
from utils import ReflectionPadding2D
import argparse

parser = argparse.ArgumentParser(description='Fast Neural style transfer with Keras.')
parser.add_argument("style_name", type=str, help='Exact name of the style (without the suffix (.jpg\.PNG))')
parser.add_argument("--col", type=int, default=188, help='image columns')
parser.add_argument("--row", type=int, default=336, help='image rows')

args = parser.parse_args()

''' Attributes '''
style_name = str(args.style_name)
img_shape = (int(args.row), int(args.col))

generator = load_model('weights/'+style_name+'.h5', custom_objects={'ReflectionPadding2D': ReflectionPadding2D})

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
while(ret):
    frame = cv2.resize(frame, img_shape)[None]
    pred = generator.predict_on_batch(frame).astype('uint8')[0]
    pred = cv2.resize(pred, (640, 480))
    cv2.imshow('style transfer', pred)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()