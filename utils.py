import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_outliers(df):
    num_cols = df.shape[1]

    fig, axes = plt.subplots(num_cols, 1, figsize=(8, 4 * num_cols), sharex=True)

    for i, column in enumerate(df.columns):
        # Detect outliers using IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

        # Plot KDE with outliers highlighted
        sns.kdeplot(df[column], ax=axes[i], label=column, shade=True)
        axes[i].scatter(x=df[column][outliers], y=np.zeros_like(df[column][outliers]),
                        color='red', marker='o', label=f'{column} Outliers')
        axes[i].set_title(f'{column} KDE Plot with Outliers')
        axes[i].set_ylabel('Density')
        axes[i].legend()

    plt.xlabel('Values')
    plt.tight_layout()
    plt.show()

