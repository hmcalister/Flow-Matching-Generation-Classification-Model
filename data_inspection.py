# %% Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% History Inspection

df = pd.read_csv("models/CIFAR10/experiment_00_mask/history.csv")
t = np.arange(len(df))

plt.figure(figsize=(10, 6))
plt.plot(t, df["train_loss"], "o-", color="k", label="Training Loss")
plt.plot(t, df["validation_loss"], "o-", color="b", label="Validation Loss")
plt.grid()
plt.legend()
plt.title("Loss")
plt.tight_layout()
# plt.savefig("figures/loss.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, df["validation_classification_accuracy"], "o-", color="b")
plt.ylim(0.5, 1)
plt.grid()
plt.title("Validation Classification Accuracy")
plt.tight_layout()
# plt.savefig("figures/classification_accuracy.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, df["validation_cross_entropy"], "o-", color="b")
plt.grid()
plt.title("Validation Classification Cross Entropy")
plt.tight_layout()
# plt.savefig("figures/cross_entropy.png")
plt.show()
