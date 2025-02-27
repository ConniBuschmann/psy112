---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
myst:
  substitutions:
    ref_test: 1
---

# <i class="fa-solid fa-dice"></i> Resampling Strategies
In the world of data science and machine learning, having access to large datasets is often a luxury rather than the norm. However, to build robust predictive models, we need effective ways to assess model performance and minimize the risk of overfitting. This is where resampling methods come into play.

Resampling methods involve **repeatedly drawing samples** from an available dataset to simulate independent datasets. These simulated sets help assess how well a trained model will perform on future data. While splitting data into training and test sets is a common practice, the results can vary depending on how the split is made. Resampling techniques provide a more reliable way to estimate model performance, reducing variability and improving accuracy.

The two mostly used resampling methods are:
- **Cross Validation** - creates non-overlapping substests that can be used to estimate the test error assocciated with a model
- **Bootstrapping** - samples with replacement, resulting in (partly) overlapping samples

```{admonition} Summary
:class: hint

Resampling methods involve:
1. Repeatedly drawing a sample from an existing dataset
2. fit the model to all resulting subsets and predict a held out amount of data
3. Examine all of the refitted models and draw appropriate conclusions
```
```{margin}
Hyperparameter are paramateres that are not learned from the data but set by the scientist before the training process begin. T
```

## Cross-Validation
### *Todays data - Iris dataset*
Let's look at how to apply the validation set approach using data.
The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

```{code-cell} 
# import packages
from sklearn import datasets
import matplotlib.pyplot as plt

# load dataset
iris= datasets.load_iris()

# Lets visualize two of our features to get an impression of the data
fig, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
fig= ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
```
The goal of the algorithm is to classify the flowers based on our features. As we only have 150 datapoints for this prediction, we can use resampling methods to avoid overfitting and get a more stable result.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz
display_quiz('Quiz/Quiz_Iris.json')
```

### Validation Set approach
In the simplest approach of cross-validation the dataset is **randomly split into two independent subsets**:
- *Training Set*: Used to train the model by learning patterns and relationships in the data
- *Validation Set*: Used to assess model performance across different models and hyperparameter choices. It therefore provides an estimation of the test error.

```{code-cell} ipython3
:tags: [remove-input]
## creating a nice looking figure to visualize the spliiting in validation set approach
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Whole dataset
ax.add_patch(patches.Rectangle((1, 4), 8, 1, color='gray', alpha=0.6))
ax.text(5, 4.5, "Whole Data Set", ha='center', va='center', fontsize=12, color='black')

# Validation set
ax.add_patch(patches.Rectangle((1, 2), 4, 1, color='lightcoral', alpha=0.6))
ax.text(3, 2.5, "Validation Set", ha='center', va='center', fontsize=12, color='black')

# Training set
ax.add_patch(patches.Rectangle((5, 2), 4, 1, color='lightblue', alpha=0.6))
ax.text(7, 2.5, "Training Set", ha='center', va='center', fontsize=12, color='black')

# Arrow from Whole Data Set to Training/Validation Set
ax.annotate('', xy=(3, 4), xytext=(3, 3), arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('', xy=(7, 4), xytext=(7, 3), arrowprops=dict(arrowstyle='->', color='black'))
# to Micha: Arrows even needed??

# Title
ax.text(5, 5.5, "Validation Set Approach", ha='center', va='center', fontsize=14, fontweight='bold')

plt.show()
```
Lets try this with our Iris dataset.

1.  As a first step, we need to **define the Target(y) and Features(X)**.

```{code-cell}
# thanks to Scikit-Learn, the Iris dataset is already predefined and consists of defined Features and Target, which we now can use 
X, y = datasets.load_iris(return_X_y=True) 
```
2. **Split** the data into training and test sample

```{code-cell}
from sklearn.model_selection import train_test_split

# sample  a training set while holding out 40% of the data for testing the classifier
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.4, random_state=0)

```



```{code-cell} ipython3
:tags: [remove-input]
<iframe src="https://trinket.io/embed/python3/09d06157a6" width="100%" height="600" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>```

```