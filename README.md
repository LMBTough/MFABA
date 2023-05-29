# MFABA

Implementation of [MFABA: A More Faithful and Accelerated Boundary-based Attribution Method for Deep Neural Networks]

## Setup

Run `pip install -r requirements.txt` to install the dependencies.

```
captum==0.6.0
matplotlib==3.6.2
numpy==1.23.3
opencv_python_headless==4.7.0.68
pandas==1.5.2
Pillow==9.4.0
torch==1.12.1+cu113
torchvision==0.13.1+cu113
tqdm==4.64.1
```

## Compute Attribution

Complete examples are shown in `example.ipynb`.Here are some sample code.

```python
from saliency.saliency_zoo import mfaba_sharp

# Load your model
model = load_model(...)
model.to(device)
model.eval()

# Load your data
img_batch = torch.load("data/img_batch.pt") # img_batch.shape = (1000,3,224,224)
target_batch = torch.load("data/label_batch.pt") # target_batch.shape = (1000,)

# Set batch_size
batch_size = 128
attributions = [] # init attributions

# Caculate attribution
for i in range(0, len(img_batch), batch_size):
    img = img_batch[i:i+batch_size].to(device)
    target = target_batch[i:i+batch_size].to(device)
    attributions.append(mfaba_sharp(model, img, target))
if attributions[0].shape.__len__() == 3:
    attributions = [np.expand_dims(attribution, axis=0) for attribution in attributions]
attributions = np.concatenate(attributions, axis=0)
```
