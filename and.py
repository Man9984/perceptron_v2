from utils.model import Perceptron
from utils.all_utils import prepare_data, create_df
import pandas as pd


AND = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
    }
OR = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,1,1,1]
    }

XOR = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,1,1,0]
    }

X,y = prepare_data(create_df(AND))
epochs = 15
eta = 0.3 # it will range in between 0 to 1
model = Perceptron(eta = eta,epochs= epochs) # [-0.00472965  0.17996672  0.03407353]
model.predict(X)
print("_____"*25)
print(X.values)
print("_____"*25)
model.fit(X,y)
model.total_loss()
model.save_model(model,"XOR.model")




