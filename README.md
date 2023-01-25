# MA2BA
Implementation of [MA2BA: A More Accurate and Accelerated Boundary-based Attribution method for Deep Neural Networks]

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

Complete examples are shown in `ma2ba.ipynb`.Here are some sample code.

```python
from methods import ma2ba_sign_after_softmax_pipeline

# Load your model
model = load_model(...)
model.to(device)
model.eval()

# Load your data, data_min is the minimum value of the image in your dataset, datamax is the maximum value of the image in your dataset
dataloader, data_min, data_max = load_test_dataset()

# Caculate attribution
for batch_x,batch_y in dataloader:
    batch_x.to(device)
    batch_y.to(device)
    attribution_map,success = ma2ba_sign_after_softmax_pipeline(model, batch_x, batch_y,data_min,data_max)
```


## Citations
```
@misc{
      title={MA2BA: A More Accurate and Accelerated Boundary-based Attribution method for Deep Neural Networks}, 
      author={Zhiyu Zhu and Jiayu Zhang},
      year={2023}
}
```










