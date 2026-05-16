from pyhere import here
import torch
from tabicl import TabICLClassifier
from tabicl import TabICLRegressor

## convert classifier from downloaded cache
model = TabICLClassifier(n_estimators=4, device="cpu")
model.model_path = "../../.cache/torch/TabICL2/tabicl-classifier-v2-20260212.ckpt"
model._load_model()

m = model.model_.state_dict()
converted = {}
for nm, par in m.items():
    converted.update([(nm, par.clone())])
      
fpath = here("tabicl-classifier-v2-20260212.pt")
torch.save(converted, fpath, _use_new_zipfile_serialization=True)



## convert regressor from downloaded cache
model = TabICLRegressor(n_estimators=4, device="cpu")
model.model_path = "../../.cache/torch/TabICL2/tabicl-regressor-v2-20260212.ckpt"
model._load_model()
m = model.model_.state_dict()
converted = {}
for nm, par in m.items():
    converted.update([(nm, par.clone())])
      
fpath = here("tabicl-classifier-v2-20260212.pt")
torch.save(converted, fpath, _use_new_zipfile_serialization=True)