import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import gan_tools as gt
import importlib
importlib.reload(gt)

import pdb

## LogisticRegression

model.class_weight = {0:100/3100, 1:3000/3100}
model  =  model.fit(x_train, y_train)

y_pred_pro = model.predict_proba(x_test)
y_pred_pro = model.predict_proba(x_load)

y_pred_cls = model.predict(x_test)
res    = model.score(x_test, y_test)


## RandomForestClassifier
rfc    = RandomForestClassifier(n_estimators=1000)
rfc    = rfc.fit(x_load, y_load)
y_pred = rfc.predict_proba(x_load)
result = rfc.score(x_test, y_test)
result