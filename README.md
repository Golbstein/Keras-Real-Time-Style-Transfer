# Real Time Style Transfer with Keras

### In this repo I've implemented the fastai-2018 style transfer in Keras

You can train and run the model in real-time with ~30fps on NVIDIA-MX150 GPU

## How to run demo:

```
git clone https://github.com/Golbstein/Keras-Real-Time-Style-Transfer.git
cd Keras-Real-Time-Style-Transfer
python real_time_neural_style.py armchair
```
or: `python real_time_neural_style.py picaso`


## Dependencies
* Python 3.6
* Keras>2.2.x

## Results of multi-loss

* Dataset: Pascal VOC 2012

![alt text](https://github.com/Golbstein/EDSR-Keras/blob/master/cat.png)

![alt text](https://github.com/Golbstein/EDSR-Keras/blob/master/bird.png)
