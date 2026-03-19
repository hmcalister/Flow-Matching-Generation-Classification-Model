# Generation / Classification Dual Model

Looking at models that map R^{data_dimension + class_dimension} -> R^{data_dimension + class_dimension}

### Experiment 00
Training on MNIST with paths of (image, noise) -> (image, label) and (noise, label) -> (image, label).

### Experiment 01
Training on MNIST with paths of (image, noise) -> (image, label) and (noise, label) -> (image, label). Now including occuluded images in both training and validation datasets.

### Experiment 02
Training on MNIST with paths of (image, noise) -> (image, label) and (noise, label) -> (image, label). Now including occuluded images in only validation datasets.
