import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv('csvs/FS.csv')
df2 = pd.read_csv('csvs/GFS.csv')
df3 = pd.read_csv('csvs/SFS.csv')
metric = 'stress'

df1['setup'] = 'Default'
df2['setup'] = 'Gradient'
df3['setup'] = 'Scalable'

df = pd.concat([df1, df2, df3], ignore_index=True)
datasets = df['dataset'].unique()

sns.set(style="whitegrid", palette="Set2", rc={"axes.facecolor": "#f0f0f0", "grid.color": "#dcdcdc", "grid.linestyle": "--"})

fig, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(20, 6), sharey=True)

boxplot_colors = sns.color_palette("Set2", n_colors=3)

for i, dataset in enumerate(datasets):
    dataset_df = df[df['dataset'] == dataset]
    
    sns.boxplot(
        x='setup',
        y=metric,
        data=dataset_df,
        ax=axes[i],
        width=0.8,
        linewidth=0.5,
        flierprops=dict(markerfacecolor='red', marker='o', markersize=1),
        hue='setup',
        palette=boxplot_colors,
        legend=False
    )

    label_name = dataset[:4]
    if len(dataset) > 4:
        label_name += '.'
    axes[i].set_title(label_name, fontsize=20)
    axes[i].set_xlabel(None)
    axes[i].set_ylabel(None)
    axes[i].set_xticklabels([])
    axes[i].grid(True, linestyle='-', alpha=0.6, axis='y')
    axes[i].tick_params(axis='y', labelsize=20)

    for spine in axes[i].spines.values():
        spine.set_visible(False)

    max_stress = df[metric].max()
    axes[i].set_ylim(bottom=0, top=max_stress * 1.1)

plt.subplots_adjust(wspace=0.1)
plt.tight_layout()

plt.show()

