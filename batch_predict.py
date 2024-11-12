import argparse
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def tsne_plot(x1, y1):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth=1, alpha=0.8,
                label='Non Fraud')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth=1, alpha=0.8,
                label='Fraud')
    plt.legend(loc='best')
    plt.title("t-SNE Visualization of Fraud and Non-Fraud Transactions")
    plt.savefig("tsne_plot.png")
    plt.close()

def main(data_path):
    df = pd.read_csv(data_path)

    print("Shape of dataset:", df.shape)
    print("Dataset info:")
    df.info()
    print("First 5 rows:")
    print(df.head())
    print("Dataset description:")
    print(df.describe())

    os.makedirs("plots", exist_ok=True)

    columns_to_graph = df.columns.tolist()
    num_columns = len(columns_to_graph)
    cols = 5
    rows = math.ceil(num_columns / cols)

    plt.figure(figsize=(15, rows * 3))
    for i, col in enumerate(columns_to_graph, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribucion: {col}')
        plt.tight_layout()
        plt.savefig(f"plots/histogram_{col}.png")
    plt.close()

    X = df.drop('Class', axis=1)
    Y = df['Class']
    tsne_plot(X, Y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA and model training")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV file")
    args = parser.parse_args()
    main(args.data_path)
