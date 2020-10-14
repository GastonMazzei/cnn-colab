import seaborn as sns
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse

with open('results.pkl','rb') as f: data=pickle.load(f)
ytrue,ypred = data['ytrue_test'], data['ypred_test']
error = mse(ytrue,ypred)/np.mean(ytrue)
data= {'angle':np.concatenate([ytrue[:,0],ypred[:,0]],0), 'scaling':np.concatenate([ytrue[:,1],ypred[:,1]],0), 'type':['true']*len(ytrue)+['predicted']*len(ypred)}
df = pd.DataFrame(data)
sns.lineplot(x='scaling',y='angle',data=df,hue='type',alpha=0.5)
plt.title(f'Relative Error was {100*round(error,2)}%')
plt.show()

