# %% Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% History Inspection

df = pd.read_csv("models/experiment_01/history.csv")
t = np.arange(len(df))

plt.figure(figsize=(10, 6))
plt.plot(t, df["train_loss"], "o-", color="k", label="Training Loss")
plt.plot(t, df["validation_loss"], "o-", color="b", label="Validation Loss")
plt.legend()
plt.title("Loss")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, df["validation_classification_accuracy"], "o-", color="b")
plt.ylim(0.9, 1)
plt.title("Validation Classification Accuracy")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, df["validation_cross_entropy"], "o-", color="b")
plt.title("Validation Classification Cross Entropy")
plt.show()
