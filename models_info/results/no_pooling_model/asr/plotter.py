import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data construction
data = []

# Eps: 1/255
data.append({'Eps': '1/255', 'Augmentation': 'With Augmentation', 'Model': 'Normal Model', 'ASR': 15.47})
data.append({'Eps': '1/255', 'Augmentation': 'With Augmentation', 'Model': 'Contrastive Model', 'ASR': 14.41})
data.append({'Eps': '1/255', 'Augmentation': 'With Augmentation', 'Model': 'Adversarial Model', 'ASR': 4.40})

data.append({'Eps': '1/255', 'Augmentation': 'No Augmentation', 'Model': 'Normal Model', 'ASR': 34.13})
data.append({'Eps': '1/255', 'Augmentation': 'No Augmentation', 'Model': 'Contrastive Model', 'ASR': 42.62})
data.append({'Eps': '1/255', 'Augmentation': 'No Augmentation', 'Model': 'Adversarial Model', 'ASR': 5.13})

# Eps: 2/255
data.append({'Eps': '2/255', 'Augmentation': 'With Augmentation', 'Model': 'Normal Model', 'ASR': 34.21})
data.append({'Eps': '2/255', 'Augmentation': 'With Augmentation', 'Model': 'Contrastive Model', 'ASR': 31.31})
data.append({'Eps': '2/255', 'Augmentation': 'With Augmentation', 'Model': 'Adversarial Model', 'ASR': 9.60})

data.append({'Eps': '2/255', 'Augmentation': 'No Augmentation', 'Model': 'Normal Model', 'ASR': 64.30})
data.append({'Eps': '2/255', 'Augmentation': 'No Augmentation', 'Model': 'Contrastive Model', 'ASR': 72.09})
data.append({'Eps': '2/255', 'Augmentation': 'No Augmentation', 'Model': 'Adversarial Model', 'ASR': 10.92})

# Eps: 4/255
data.append({'Eps': '4/255', 'Augmentation': 'With Augmentation', 'Model': 'Normal Model', 'ASR': 66.69})
data.append({'Eps': '4/255', 'Augmentation': 'With Augmentation', 'Model': 'Contrastive Model', 'ASR': 62.74})
data.append({'Eps': '4/255', 'Augmentation': 'With Augmentation', 'Model': 'Adversarial Model', 'ASR': 18.79})

data.append({'Eps': '4/255', 'Augmentation': 'No Augmentation', 'Model': 'Normal Model', 'ASR': 92.11})
data.append({'Eps': '4/255', 'Augmentation': 'No Augmentation', 'Model': 'Contrastive Model', 'ASR': 95.96})
data.append({'Eps': '4/255', 'Augmentation': 'No Augmentation', 'Model': 'Adversarial Model', 'ASR': 22.81})

# Eps: 8/255
data.append({'Eps': '8/255', 'Augmentation': 'With Augmentation', 'Model': 'Normal Model', 'ASR': 94.28})
data.append({'Eps': '8/255', 'Augmentation': 'With Augmentation', 'Model': 'Contrastive Model', 'ASR': 91.11})
data.append({'Eps': '8/255', 'Augmentation': 'With Augmentation', 'Model': 'Adversarial Model', 'ASR': 37.70})

data.append({'Eps': '8/255', 'Augmentation': 'No Augmentation', 'Model': 'Normal Model', 'ASR': 99.00})
data.append({'Eps': '8/255', 'Augmentation': 'No Augmentation', 'Model': 'Contrastive Model', 'ASR': 99.81})
data.append({'Eps': '8/255', 'Augmentation': 'No Augmentation', 'Model': 'Adversarial Model', 'ASR': 45.61})

# Eps: 16/255
data.append({'Eps': '16/255', 'Augmentation': 'With Augmentation', 'Model': 'Normal Model', 'ASR': 99.52})
data.append({'Eps': '16/255', 'Augmentation': 'With Augmentation', 'Model': 'Contrastive Model', 'ASR': 98.08})
data.append({'Eps': '16/255', 'Augmentation': 'With Augmentation', 'Model': 'Adversarial Model', 'ASR': 59.50})

data.append({'Eps': '16/255', 'Augmentation': 'No Augmentation', 'Model': 'Normal Model', 'ASR': 99.73})
data.append({'Eps': '16/255', 'Augmentation': 'No Augmentation', 'Model': 'Contrastive Model', 'ASR': 99.97})
data.append({'Eps': '16/255', 'Augmentation': 'No Augmentation', 'Model': 'Adversarial Model', 'ASR': 65.90})

df = pd.DataFrame(data)

# Set plot style
sns.set_theme(style="whitegrid")

# Create the grouped bar plot using catplot
g = sns.catplot(
    data=df, 
    x='Eps', 
    y='ASR', 
    hue='Model', 
    col='Augmentation',
    kind='bar',
    height=6, 
    aspect=1.2,
    palette="viridis",
    legend_out=True
)

# Customizing the axes and titles
g.set_axis_labels("Epsilon (Perturbation Magnitude)", "Attack Success Rate (%)")
g.set_titles("{col_name}")
g.set(ylim=(0, 110))

# Add value labels on top of bars
for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, rotation=90, fontsize=9)

# Adjust layout
plt.subplots_adjust(top=0.85)
g.fig.suptitle('PGD Attack Success Rate by Epsilon and Augmentation Strategy', fontsize=16)

# Save the plot
plt.savefig('pgd_asr_barplot.png')