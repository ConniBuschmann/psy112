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

## Cross-Validation
### Validation Set approach
In the simplest approach of cross-validation the dataset is **randomly split into two independent subsets**:
- *Training Set*: Used to train the model by learning patterns and relationships in the data
- *Validation Set*: Used to assess model performance across different models and hyperparameter choices 

```{margin}
Hyperparameter are paramateres that are not learned from the data but set by the scientist before the training process begin. They are essential for fine-tuning models to achieve optimal performance
```