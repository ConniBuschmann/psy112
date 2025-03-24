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

<iframe src="https://trinket.io/embed/python3/12222a549d51" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

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


```{admonition} Handling big data
:class: hint

To handle large datasets efficiently in linear modeling, three key techniques are used:

- **Subset Selection**
- **Dimension Reduction**
- **Regularization / Shrinkage**
```


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

# Prepare data - defining target and features
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

In the following example, we evaluate all possible feature combinations from 1 to 9 predictors and identify the best subset.

```{code-cell} 
# Preperation 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from mlxtend.feature_selection import ExhaustiveFeatureSelector


# Define target and features
X = hitters_subset.drop(columns=["Salary"])
y = hitters_subset["Salary"]

# Define the regression model 
model = LinearRegression()

# Use 5-fold cross-validation to evaluate model performance
cv_folds = 5
```
Before performing the best subset selection, we split our data into training and test dataset. Although the selection function uses cross-validation to identify the best subset of predictors (Step 3), this evaluation is done during the selection process and can still overfit to the data. To fairly assess how the final model performs on new data, we split off a test set and use it only after feature selection is complete.

|Purpose                        	   | What is it for?                                   | When?                                               |
|------------------------------------- |---------------------------------------------------|-----------------------------------------------------|
|Cross Validation in selection function|Helps choose the best subset of features           |During selection                                     |
|Test Set Evaluation                   |Checks how well the final model performs           |After selection                                      |


```{code-cell} 
# Split data BEFORE selection
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```{code-cell} 
# Perform best subset selection 
efs = ExhaustiveFeatureSelector(model, 
          min_features=1, 
          max_features=9,                        # Try all subsets from 1 to 9 features
          scoring='neg_mean_squared_error',      # trainings MSE
          cv=cv_folds,                           # Apply k-fold cross-validation (e.g., 5-fold)
          print_progress=False)

# fit it only on trainings data!
efs.fit(X_train, y_train)

# Output the best feature subset and the corresponding cross-validated R^2 score
print("Best Features:", efs.best_feature_names_)
# flipping the sign of MSE, because less negative is the best
MSE= -efs.best_score_ 
print("Best Trainings MSE:", MSE)
```
At this point, we've completed Step 2 of our model selection process: For each model size (1 to 9 predictors), we identified the best-performing combination.
Now, we can take a closer look at all the models that were evaluated, not just the best one.

To better understand the trade-off between model complexity and predictive power, we can visualize the best model performance at each model size.

```{code-cell}
import matplotlib.pyplot as plt

# Extract metric results
metric_dict = efs.get_metric_dict()
results = []

for subset in metric_dict.values():
    n_features = len(subset['feature_idx'])
    avg_mse = -subset['avg_score']  # flip the sign
    results.append((n_features, avg_mse))

# Create DataFrame
results_df = pd.DataFrame(results, columns=["n_features", "avg_mse"])

# Aggregate by number of features (in case multiple subsets have the same size)
results_df = results_df.groupby("n_features", as_index=False).min()

# Plot
plt.figure(figsize=(8, 5))
plt.plot(results_df["n_features"], results_df["avg_mse"], marker="o")
plt.title("Best Subset Selection: MSE vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("Cross-validated MSE")
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

So let`s apply forward stepwise selection. 

After defining the predictors, response variable, and the number of cross-validation folds, we can now run the forward stepwise selection. Unlike exhaustive selection, which fits all possible models, forward selection starts with no predictors and adds one feature at a time — always choosing the one that improves performance the most. This process continues until we reach the maximum number of features (in this case, 9).If you'd rather stop after a specific number of features, you can control that using the `k_features` parameter.

```{code-cell}
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection  import train_test_split

# Run forward selection using MSE as the scoring metric
forward = SequentialFeatureSelector(
    model,              # defined model 
    k_features=(1,9),   # Stopping criteria: Try models with 1,2,..,9 features
    forward=True,       # use forward selection 
    floating=False,     # Do not use floating selection -> Classic forward selection
                        # floating =True : after each addition, it also checks if it should remove a feature that has become less useful
    scoring='neg_mean_squared_error',       
    cv=cv_folds)

# Fit the prepared model on our trainings data
sfs = forward.fit(X_train, y_train)
```
Next we extract the final selected subset of features, which the algorithm determined to be the most predictive of `Salary` under forward selection.

```{code-cell}
# Get selected features
selected_features = list(sfs.k_feature_names_)
print("Selected features:", selected_features)

# Flip the sign of the best score to get the actual MSE
best_cv_mse = -sfs.k_score_
print(f"Cross-validated MSE of best model: {best_cv_mse:.2f}")
```

Just like we did with best subset selection, we now visualize the R² score at each model size. This helps us understand how the model performance improves as we add more predictors.

```{code-cell} ipython3
:tags: [remove-input]

# Plot the metrics for each number of features using MSE
import matplotlib.pyplot as plt
import pandas as pd

# Access MSE for each step (stored during selection)
mse_scores = []
num_features = []

for k in sfs.get_metric_dict().keys():
    neg_mse = sfs.get_metric_dict()[k]['avg_score']  # Still negative
    mse = -neg_mse                                    # Flip to positive MSE
    mse_scores.append(mse)
    num_features.append(len(sfs.get_metric_dict()[k]['feature_idx']))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(num_features, mse_scores, marker='o')
plt.title("Forward Selection: MSE vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("Trainings MSE")
plt.show()
```

<iframe src="https://trinket.io/embed/python3/a00b2117cbe7" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>
<br>



```{admonition} Interim Summary
:class: tip

- As we can see, we ended up with the same three predictors as in best subset selection - `CATBat`, `Hits`, `CRuns`, `HmRun`. Once again, a combination of these three seems to perform best in predicting `Salary`. 
- However, this is not necessarily always the case — best subset and stepwise selection can, and often do, **result in different predictors** or even a different number of predictors being selected.

```

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

#### What next?
Once we have identified the features that are relevant for predicting the outcome, let`s evaluate the model performance and estimate true test error with the 4 predictors identified by Best Subset Selection and Forward Stepwise Seletion.
```{code-cell}
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression

selected_features = ['CAtBat', 'CRuns', 'Hits', 'HmRun'] 

# use splitted data only from selected features
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# fit model with selected featues
model = LinearRegression()
model.fit(X_train_sel, y_train)

# Predict on the test set
y_pred = model.predict(X_test_sel)


# Evaluate
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred)

# Print results
print(f"Test MSE: {mse_test:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")
print(f"Test R²: {r2_test:.4f}")
```
What does this mean for our prediction using the 4 features?
- On average, our predictions deviate from the actual salary by about $400.
- Our model explains only a small portion (~11%) of the variability in salary. That also means that te majority of factors influencing salary are not captured by these predictors.

### Dimension Reduction
Dimensionality reduction is a model selection technique that simplifies high-dimensional datasets by transforming them into a smaller set of uncorrelated components. Instead of selecting individual features, it **combines correlated variables into new components** that retain most of the data’s variance. This reduces computational cost, lowers the risk of overfitting, and improves model performance — especially when many features are present. The first component captures the most variance, the second the next most, and so on. This method helps preserve patterns and trends in the data while working with fewer, more manageable inputs.