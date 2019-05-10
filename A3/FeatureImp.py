from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def feature_weight():
    df = pd.read_csv("processed.cleveland.data", header=None,
                          names=['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral in mg/dl',
                                 ' fasting blood sugar', 'resting electrocardiographic results',
                                 ' maximum heart rate achieved', ' exercise induced angina', 'oldpeak',
                                 'the slope of the peak exercise ST segment',
                                 'number of major vessels (0-3) colored by flourosopy', 'thal', 'target'])

    df.replace('?', np.nan, inplace=True)  # Replace ? values
    df = df.dropna()

    df.loc[df['target'] != 0] = 1
    total_features = df.columns.difference(['target'],sort=False)
    features = df[total_features].values
    labels = df["target"].values

    rfc = RandomForestClassifier(n_estimators=5000,n_jobs=5)
    rfc.fit(features, labels)
    importances = rfc.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Features' weight are followed:")

    for i in range(features.shape[1]):
        print("%d. feature %d (%f ->" % (i + 1, indices[i], importances[indices[i]]) + " " + total_features[
            int(indices[i])] + ")")

    plt.title('Feature Importance')
    plt.figure(figsize=(20, 15))
    plt.bar(range(features.shape[1]), importances[indices], color="red", align="center")
    plt.xticks(range(features.shape[1]), indices)
    plt.xlim([-1, features.shape[1]])
    plt.xlabel("Feature's index")
    plt.ylabel("Weight")
    plt.savefig('Feature_importance.jpg')
    return total_features[indices[:8]]


if __name__ == "__main__":
    feature_weight()