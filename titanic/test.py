import pickle
import pandas as pd

with open('D:\\AI\\Machine Learning\\titanic\\model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

data = {
    'pclass': [3],
    'sex': ['female'],
    'age': [0.166667],
    'fare': [0.0000],
    'embarked': ['S'],
    'who': ['woman'],
    'adult_male': [False],
    'family_size': [5]
}
test = pd.DataFrame(data)

res = loaded_model.predict(test)
if(res[0]==1):
    print("Survived")
else: ("Died")