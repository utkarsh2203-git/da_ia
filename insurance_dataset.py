from google.colab import files
uploaded = files.upload()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('insurance.csv')
df.head(10)
df.info()
df.isnull().sum()
df.duplicated().sum()
df = df.drop_duplicates()

df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'])

plt.scatter(df['bmi'], df['charges'], c=df['smoker'])
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.title('BMI vs Charges')
plt.show()

df[['age','bmi','children','charges']].head(10)

sns.histplot(df['charges'], kde=True)
plt.title('Charges Distribution')
plt.show()

df['sex'] = df['sex'].map({'female':0,'male':1})
corr = df.corr()
print(corr)

sns.heatmap(corr, annot=True)
plt.title('Correlation Heatmap')
plt.show()
