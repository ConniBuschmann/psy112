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

# <i class="fa-brands fa-python"></i> Model Selection
In neurocognitive psychology, brain imaging (e.g., fMRI, EEG), cognitive assessments, and behavioral experiments generate vast datasets, measuring thousands of brain regions, connectivity patterns, and behavioral traits. This data captures everything from patient vitals to cognitive processes and offers **detailed insights** with immense predictive power. However, it also introduces challenges: **too many predictors**, making analysis and interpretation difficult.

## The large p issue
**Big Data** refers to large datasets with many predictors, that cannot be processed or analyzed using traditional data processing techniques. For our prediction models, this brings some issues:
- While the linear model can in theory still be used for such data, the **ordinary least squares fit becomes infeasible**, especially when p > n 
- The large amount of features reduce interpretability


This is where **linear model selection** becomes essential, offering techniques to refine our models and extract meaningful insights from high-dimensional neurocognitive data!

----------------------------------------------------------------
### *Todays data with many predictors - Hitters dataset*
For pracitcal demonstration, we will use the `Hitters` dataset. This data set provides Major League Baseball Data from the 1986 and 1987 seasons. It contains 322 observations of major league players on 20 variables. The Research aim is to predict a baseball player's salary on the basis of various predictors associated with the performance in the previous year.

```{code-cell} 
# import packages
import statsmodels.api as sm 

# get dataset
hitters = sm.datasets.get_rdataset("Hitters", "ISLR").data
```
Get yourself familiar with the dataset. Look at the predictor variables. Which information do we include to predict the salary? 
You can check the variable names here: https://islp.readthedocs.io/en/latest/datasets/Hitters.html  
Also take a closer look to the variable you want to predict! Do we have the information(s) that we need for all players?

<iframe src="https://trinket.io/embed/python3/d980217b790c" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

For computationally reasons, we will not include all predictors but only a small subset
```{code-cell} 
# keeping a total of 10 variables - the outcome Salary and 9 predictors.
# Keeping only Salary and 9 predictors
hitters_subset = hitters[["Salary", "CHits", "CAtBat", "CRuns", "CWalks", "Assists", "Hits", "HmRun", "Years", "Errors"]].copy()

# make sure the rows, containing missing values, are dropped
hitters_subset.dropna(inplace=True)

hitters_subset.head()
```


Okay, now that we know our dataset, let's look at how to handle such a large number of predictors! 

----------------------------------------------------------------
## Handling big data in linear models
To handle large datasets efficiently in linear modeling, three key techniques are used:
- Subset Selection
- Dimension Reduction
- Regularization/Shrinkage 


By leveraging these methods, we can build robust predictive models that remain efficient and interpretable, even in the face of Big Data challenges.


### Subset Selection
In subset selection we identify a subset of *p* predictos that are truly related to the outcome. The model get fitted using least squares on the reduces set of variables.

How do we determine which variables are relevant?! 

####  Best Subset Selection

```{image} ./figures/BestSubsetSelection.drawio.png
:alt: ModelSelection
:width: 30%
:align: left
```

```{margin}
The Null Model only predicts the sample mean
```
<br>
<br>

1. Consider all possible models
    - Starting with Null Model <em>M0</em>, which contains no predictors
    - Iteratively adding a predictor to the model
2. Identify the Best Model of each size
    - Either by the smallest RSS or the largest <code>R²</code></li>
3. Identify the Best Overall Model
    - Use cross-validation to find the best <em>Mk</em></li>

<br>
<br>
<br>

Let's get back to our dataset and see how Best Subset Seletion is performed in python.

```{code-cell} ipython3
:tags: ["remove-input"]
from jupyterquiz import display_quiz
display_quiz("Quiz/Quiz_BestSubsetSelection.json", shuffle_answers=False)
```

<br>

**To MICHA:** We could also think about using abess. https://github.com/abess-team/abess
But for now I decided to not use it since it just do everything and I thought it might be harder to understand the concept behind it, because we  . However if you decide to go that way, we can use the following code chunk.
We could also consider using abess (https://github.com/abess-team/abess) — it's a nice OPEN SOURCE library that performs best subset selection super efficiently.
However, I decided not to use it for now because it does everything automatically, and I thought that might make it harder for students to understand what's actually happening. but if you prefer to go use abess, here is the code chunk, we could use:
If you need more details:https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_1_LinearRegression.html#sphx-glr-auto-gallery-1-glm-plot-1-linearregression-py



```{code-cell} 
from abess.linear import LinearRegression 
import numpy as np
import pandas as pd

# Prepare data
X= np.array(hitters_subset.drop("Salary", axis=1))
y= np.array(hitters_subset["Salary"])

# Use abess for best subset selection
model = LinearRegression(support_size=range(1, 10))  # support_size is how many features to try
model.fit(X, y)

# Get selected features (non-zero coefficients)
ind = np.nonzero(model.coef_)
print("non-zero:\n", hitters_subset.columns[ind])
print("coef:\n", model.coef_)
```
The abess algorithm evaluates all possible combinations of our 9 predictors and automatically selects the best subset based on internal criteria (e.g., minimizing BIC). In our case, it selected just two predictors: `CAtBat` and `Assists` 

You can also implement Best Subset Selection manually using the `mlxtend` library. Unlike abess, this approach allows you to explicitly control and understand what’s happening at each step of the selection process.

In the following example, we evaluate all possible feature combinations from 1 to 9 predictors and identify the best subset based on cross-validated R² performance.

```{code-cell} 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from mlxtend.feature_selection import ExhaustiveFeatureSelector


# Define predictors (X) and response variable (y)
X = hitters_subset.drop(columns=["Salary"])
y = hitters_subset["Salary"]

# Define the regression model 
model = LinearRegression()

# Use 5-fold cross-validation to evaluate model performance
cv_folds = 5

# Perform best subset selection 
efs = ExhaustiveFeatureSelector(model, 
          min_features=1, 
          max_features=9,       # Try all subsets from 1 to 9 features
          scoring='r2',         # Use R² as performance metric
          cv=cv_folds,          # Apply k-fold cross-validation (e.g., 5-fold)
          print_progress=False)

efs.fit(X, y)

# Output the best feature subset and the corresponding cross-validated R^2 score
print("Best Features:", efs.best_feature_names_)
print("Best R² (CV):", efs.best_score_)
```
At this point, we've completed Step 2 of our model selection process: For each model size (1 to 9 predictors), we identified the best-performing combination.
Now, we can take a closer look at all the models that were evaluated, not just the best one.


```{code-cell}
import pandas as pd

# Access the CV results for every model evaluated
results = efs.get_metric_dict()

# Convert to DataFrame
summary_df = pd.DataFrame([
    {
        'features': res['feature_names'],
        'r2_mean': res['avg_score']
    }
    for res in results.values()
])

# Sort by best R²
summary_df = summary_df.sort_values(by='r2_mean', ascending=False)

summary_df.head()  # Show top models
```
This table gives us an overview of the best feature combinations, sorted by their cross-validated R². It helps us see which models performed well and how much difference the feature choice made. To better understand the trade-off between model complexity and predictive power, we can visualize the best model performance at each model size.

```{code-cell}
import matplotlib.pyplot as plt
import pandas as pd

# Extract metric results
metric_dict = efs.get_metric_dict()
results = []

for subset in metric_dict.values():
    n_features = len(subset['feature_idx'])
    avg_score = subset['avg_score']
    results.append((n_features, avg_score))

# Create DataFrame
results_df = pd.DataFrame(results, columns=["n_features", "avg_r2"])

# Aggregate by number of features (in case multiple subsets have the same size)
results_df = results_df.groupby("n_features", as_index=False).max()

# Plot
plt.figure(figsize=(8, 5))
plt.plot(results_df["n_features"], results_df["avg_r2"], marker="o")
plt.title("Best Subset Selection: R² vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("Cross-validated R²")
plt.show()

```
This plot helps us visualize how performance improves as we increase the number of features. It reflects Step 2 of our selection process and gives us insight into the bias-variance tradeoff: at some point, adding more features doesn't necessarily improve the model much.


#### Forward Stepwise Selection
Best subset selection is not feasible for very large *p* due to its computational demands. A more efficient way solving this problem, is foward stepwise selection. 

```{image} ./figures/ForwardStepwiseSelection.drawio.png
:alt: ModelSelection
:width: 30%
:align: left
```
<br>
<br>

1. Beginning with null hypothesis
2. Adding the most significant variables one after the other
    - Either by the smallest RSS or the largest <code>R²</code></li>
3. Repeat it until...
    - reaching a stopping criteria
    - k=p
4. Identifying the single best model using cross-validation

<br>
<br>

**TO MICHA**: Also here we have the oppurtunity to just use a forwardSelection function: https://github.com/talhahascelik/python_stepwiseSelection/blob/master/stepwiseSelection.py#L19 or to do it step by step by our own:

Or use from mlxtend.feature_selection import SequentialFeatureSelector:

So let`s apply forward stepwise selection. Just as before, we will use cross-validated R² to assess model performance.

```{code-cell}
import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Define predictors (X) and response variable (y)
X = hitters_subset.drop(columns=["Salary"])
y = hitters_subset["Salary"]

# Define the regression model 
model= LinearRegression()

# define k for k-fold cross validation
cv_folds = 5
```
After defining the predictors, response variable, and the number of cross-validation folds, we can now run the forward stepwise selection. Unlike exhaustive selection, which fits all possible models, forward selection starts with no predictors and adds one feature at a time — always choosing the one that improves performance the most. This process continues until we reach the maximum number of features (in this case, 9).If you'd rather stop after a specific number of features, you can control that using the `k_features` parameter.


```{code-cell}
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Run forward selection using R² as the scoring metric
forward = SequentialFeatureSelector(
    model,              # defined model 
    k_features=(1,9),   # Stopping criteria: Try models with 1,2,..,9 features
    forward=True,       # use forward selection 
    floating=False,     # Do not use floating selection -> Classic forward selection
                        # floating =True : after each addition, it also checks if it should remove a feature that has become less useful
    scoring='r2',       # Use R² score as the metric to evaluate model performance
    cv=cv_folds)

# Fit the prepared model on our data
sfs = forward.fit(X, y)
```
Next we extract the final selected subset of features, which the algorithm determined to be the most predictive of Salary under forward selection.
```{code-cell}
# Get selected features
selected_features = list(sfs.k_feature_names_)
print("Selected features:", selected_features)
```
Just like we did with best subset selection, we now visualize the R² score at each model size. This helps us understand how the model performance improves as we add more predictors.
```{code-cell}
# Plot the metrics for each number of features 
import matplotlib.pyplot as plt
import pandas as pd
 import pandas as pd

# Access R² for each step (get stroed while forward stepwise selection process)
r2_scores = []
num_features = []

for k in sfs.get_metric_dict().keys():
    r2 = sfs.get_metric_dict()[k]['avg_score']
    r2_scores.append(r2)
    num_features.append(len(sfs.get_metric_dict()[k]['feature_idx']))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(num_features, r2_scores, marker='o')
plt.title("Forward Selection: R² vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("Cross-validated R²")
plt.show()

```

<br>
--------------------------------------------------------------


**Interim Summary**
- As we can see, we ended up with the same three predictors as in best subset selection - `CRuns`, `Hits`, `Errors`. Once again, a combination of these three seems to perform best in predicting `Salary`. 
- However, this is not necessarily always the case — best subset and stepwise selection can, and often do, **result in different predictors** or even a different number of predictors being selected.
- Still, neither model explains much more than a third of the variance in salary — which suggests that **many other factors** not included in our dataset may be influencing player salaries (e.g., team dynamics, contracts, injuries, reputation, etc.).
    - In this case, both models demonstrate limited predictive power, and while R² values of 0.3–0.4 can be acceptable in some real-world contexts, they should be **interpreted with caution**.



--------------------------------------------------------------

To wrap up our subset selection methods, let’s briefly explore backward stepwise selection.


#### Backward Stepwise Selection
```{image} ./figures/BackwardStepwiseSelection.drawio.png
:alt: ModelSelection
:width: 30%
:align: left 
```

```{margin}
The Full Model contain all p predictors!
```

<br>
<br>

1. Beginning with the Full Model <em>Mp</em>
2. Iteratibely removes the least usefull predictor
3. Repeat it until...
    - reaching a stopping criteria
    - <em>k=0</em>
4. Identify the best overall model using cross-validation


<br>
<br>

Backward stepwise selection works very similarly to forward selection — the main difference is that we start with the full model and remove features one by one. 


```{code-cell} ipython3
:tags: [remove-input]

from jupytercards import display_flashcards
display_flashcards('Quiz/BackwardSelection_Flashcard.json')
```


```{admonition} Subset Selection Summary
:class: tip

| Best Subset Selection            	  | Forward Stepwise Selection                        | Backward Stepwise Selection                         |
|-------------------------------------|---------------------------------------------------|-----------------------------------------------------|
|**-** computationally very expensive |**-** not guaranteed to find best model            |**-** not guaranteed to find best model              |
|**-** with many *p* may overfit      |**+** possible to us when p is very large          |**+** possible to us when p is very large, given p<n |
|**+** able to find the best model    |**+** computationally less demanding               |**+** computationally less demanding                 |


```


```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz
display_quiz('Quiz/Quiz_SubsetSelection.json')
```