import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv('csvs/default.csv')
df2 = pd.read_csv('csvs/gradient.csv')
df3 = pd.read_csv('csvs/scalable.csv')

# Add a column identifying each setup
df1['setup'] = 'Default'
df2['setup'] = 'Gradient'
df3['setup'] = 'Scalable'

df = pd.concat([df1, df2, df3], ignore_index=True)
datasets = df['dataset'].unique()

fig, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(20, 6), sharey=True)

for i, dataset in enumerate(datasets):
    dataset_df = df[df['dataset'] == dataset]
    
    sns.boxplot(x='setup', y='stress', data=dataset_df, ax=axes[i])
    axes[i].set_title(f'{dataset}')
    #axes[i].set_xlabel('Setup')
    axes[i].set_ylabel('Stress')

plt.tight_layout()
plt.show()
