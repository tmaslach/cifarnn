*License is GPLv2*

This is my experimentation with CIFAR-10 and Neural Networks with Pytorch and
tools.

You'll need to install a few packages. You can try to use newer versions. Just
omit the ==<version> below.

```
pip install torch==1.6.0
pip install torchvision==0.7.0
pip install matplotlib==3.2.2
pip install tensorboardX==2.1
```

You can use the code right out the gate to predicut.  Just go into the cifarnn
repo and run:
```
python predict <imagefile>
```
Where <imagefile> is the image file you want my already train neural Networks
to guess.

If you want to re-train the model, run:
```
python train
```

Feel free to tweak parameters as you see fit.  I did not specify any parameters
to be passed in via argument for the train option, only because I preferred to 
tweak in code (or use optuna for hyperparameter calibration).

Nice things to do:
  - Make it so predict doesn't require matplotlib, tensorboardX
  - Get better than 92.623%
