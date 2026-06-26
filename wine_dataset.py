from google.colab import files
files.upload()

import pandas as pd
from scipy.spatial.distance import cdist, minkowski
import matplotlib.pyplot as plt
import seaborn as sns

wine=pd.read_csv('/content/wine.csv')
cols=['Alcohol','Malic','Ash','Magnesium','Phenols','Flavanoids']
data=wine[cols].head(10)

euclidean=cdist(data,data,metric='euclidean')
sns.heatmap(pd.DataFrame(euclidean),annot=True,cmap='coolwarm')
plt.title("Euclidean Distance Matrix")
plt.show()

manhattan=cdist(data,data,metric='cityblock')
sns.heatmap(pd.DataFrame(manhattan),annot=True,cmap='coolwarm')
plt.title("Manhattan Distance Matrix")
plt.show()

A,B,C=data.iloc[0],data.iloc[1],data.iloc[2]
pairs=[('A-B',A,B),('A-C',A,C),('B-C',B,C)]
p_values=[1,2,3,10]
results=[]
for name,x,y in pairs:
    for p in p_values:
        results.append([name,p,minkowski(x,y,p)])

distance_df=pd.DataFrame(results,columns=['Pair','p','Distance'])
print(distance_df)

for pair in distance_df['Pair'].unique():
    subset=distance_df[distance_df['Pair']==pair]
    plt.plot(subset['p'],subset['Distance'],marker='o',label=pair)
plt.xlabel("p value")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance vs p")
plt.legend()
plt.show()
