import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

def prepare_data(df):
    X = df.drop("y", axis=1)
    y = df["y"]
    print("2. Prepare data train and Target")
    return X, y

def create_df(dict):
    print("1. Dataframe Created")
    return pd.DataFrame(dict)


def save_model(self,model,filename):
        model_dir = "Models"
        os.makedirs(model_dir,exist_ok=True)
        file_path = os.path.join(model_dir,filename)
        joblib.dump(model,file_path)


def save_plot(df, file_name, model):
    def _create_base_plot(df):
        df.plot(kind='scatter', x = "x1", y = "x2", color = "y", s = 100, cmap = "winter")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(y=0, color="black", linestyle="--", linewidth=1)
        figure = plt.gcp()  # get current figure.
        figure.set_size_inches(10, 8) # shape the size of inches

    def _plot_decision_regions(X, y, classifier, resolution=0.02):
        color = ("red","blue","lightgreen", "gray", "cyan")
        cmap = plt.color.ListedColormap(color[:,len(np.unique(y))])
        X = X.values()
        x1 = X[:, 0]
        x2 = X[:, 1]
        x1_min, x1_max = x1.min() - 1, x1.min() + 1
        x2_min, x2_max = x2.min() - 1, x2.min() + 1

        xx1, xx2 =np.meshgrid(np.arange(x1_min, x1_max,resolution),
                              np.arange(x2_min, x2_max,resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = np.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha= 0.2 , cmap = cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.plot()

        pass
    X,y = prepare_data(df)
    _create_base_plot(df)
    _plot_decision_regions(X,y, model)

    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, file_name)
    plt.savefig(plot_path)
