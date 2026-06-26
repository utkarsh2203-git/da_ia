from google.colab import files
files.upload()

import pandas as pd
import numpy as np

nutrient=pd.read_csv('/content/nutrient.csv')

nutrient['energy_group']=pd.qcut(nutrient['energy'],3,labels=['Low','Medium','High'])
nutrient['fat_group']=pd.qcut(nutrient['fat'],3,labels=['Low','Medium','High'])

item1=nutrient.iloc[0]
item2=nutrient.iloc[1]
match=((item1['energy_group']==item2['energy_group'])+(item1['fat_group']==item2['fat_group']))
print("Simple Matching Similarity:",match/2)

nutrient['high_protein']=(nutrient['protein']>nutrient['protein'].median()).astype(int)
nutrient['high_iron']=(nutrient['iron']>nutrient['iron'].median()).astype(int)

x=nutrient.iloc[0][['high_protein','high_iron']]
y=nutrient.iloc[1][['high_protein','high_iron']]
z=nutrient.iloc[2][['high_protein','high_iron']]

def jaccard(a,b):
    inter=np.sum((a==1)&(b==1))
    union=np.sum((a==1)|(b==1))
    return inter/union if union else 0

def smc(a,b):
    return np.sum(a==b)/len(a)

print("Jaccard (x,y):",jaccard(x,y))
print("SMC (x,y):",smc(x,y))
print("Jaccard (x,z):",jaccard(x,z))
print("SMC (x,z):",smc(x,z))
print("Jaccard (y,z):",jaccard(y,z))
print("SMC (y,z):",smc(y,z))
