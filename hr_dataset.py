from google.colab import files
files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

df = pd.read_csv('HR_comma_sep.csv')
cols=['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company']
data=df[cols]

mean=data.mean()
median=data.median()
mode=data.mode().iloc[0]
range_val=data.max()-data.min()
variance=data.var()
std_dev=data.std()
iqr=data.quantile(0.75)-data.quantile(0.25)

summary=pd.DataFrame({
'Mean':mean,
'Median':median,
'Mode':mode,
'Variance':variance,
'Std Dev':std_dev,
'IQR':iqr
})
print(summary)

sns.histplot(df['satisfaction_level'],kde=True)
plt.show()

sns.boxplot(x=df['satisfaction_level'])
plt.show()

stats.probplot(df['satisfaction_level'],dist='norm',plot=plt)
plt.show()

sns.pairplot(df,hue='left',vars=['satisfaction_level','last_evaluation','number_project','average_montly_hours'])
plt.show()
