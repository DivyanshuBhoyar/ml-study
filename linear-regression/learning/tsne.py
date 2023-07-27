# Importing Necessary Modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sn

# Reading the data using pandas
df = pd.read_csv("mnist_train.csv")
print(df.head(4))
labels = df["label"]
# Drop the label feature and store the pixel data in dataframe
data = df.drop("label", axis=1)

standardized_data = StandardScaler().fit_transform(data)
print(standardized_data.shape)
data_1000 = standardized_data[0:1000, :]
labels_1000 = labels[0:1000]

model = TSNE(n_components=2, random_state=0)
# configuring the parameters
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations
# for the optimization = 1000
tsne_data = model.fit_transform(data_1000)

tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sn.FacetGrid(tsne_df, hue="label", size=6).map(
    plt.scatter, "Dim_1", "Dim_2"
).add_legend()
plt.show()
