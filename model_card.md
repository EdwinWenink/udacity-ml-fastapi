# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was trained by Edwin Wenink for demo purposes.
The model is a Random Forest model (ensemble of decision trees) that was trained with all default parameters except `n_estimators=200` (`scikit-learn 1.3.0`).
No further optimization was performed.

## Intended Use

The model can be used to predict whether a person's income exceeds $50K per year based on census data (i.e. a binary classification problem).
This model is trained solely for demo purposes and requires much more work and ethical scrutiny before being used in any other context.

## Training Data

This project uses a cleaned version of the [Census Income](https://archive.ics.uci.edu/dataset/20/census+income) data set.

A simple train-test split (80-20%) was used to separate training and evaluation data from the Census Income data set.

An exploratory data analysis provided some insights about the data set:

- The original data set had white spaces in the CSV. This was preprocessed manually once.
- Missing values are registered as '?' and should be converted to NaN for easier analysis
- The marital status factor has some categories that are very similar and could be merged.
- Capital gain and loss could be combined into a netto capital value that is positive or negative
- There are 23 duplicate rows
- Correlation:
    * `education_num` and `education` are highly correlated.
      `education_num` seems to be a label-encoding of `education`. Drop either of them.
    * `sex` and `relationship` are highly correlated
      This makes sense since relationship 'Wife' implies sex 'Female' in the context of this data set, and so on.
- Imbalanced features:
    * `race`
    * `native-country`
    * `capital-gain`
    * `capital-loss`
- Missing values:
    * `workclass`: 1836 (5.6%)
    * `occupation`: 1843 (5.7%)
    * `native-country`: 483 (1.7%)
- Zeros:
    * `capital-gain` 29849 (91.7%)
    * `capital-loss` 31042 (95.3%)

Finally, there are more instance for "<=50K" (24720) than ">50K" (7841).
The classification report indeed shows the model is better at predicting the "<=50K" (encoded as 0) than ">50K" (encoded as 1).

Further preprocessing was done that addresses a few of the identified issues:

- Stripped spaces from the input csv file.
- Dropped rows with missing values, which were encoded as '?'.
- Capital gain and loss were combined in a single feature with the netto capital change.
- Outliers on capital change with a z-score > 3 were dropped.
- The column `education-num` was dropped because it was an label encoding of `education`.


## Evaluation Data

The evaluation data was generated using a simple train-test split on the Census Income data set.
The same data preparation as described for the training set were used.
Hence, the same caveats are applicable.
This evaluation data set should be used only for initial model selection.


## Metrics

This project tracks the precision, recall, and f1-score to estimate the model performance in the binary classification task.

Confusion matrix:

```
 [[4093  456]
 [ 616  838]]
```

Classification report:

```
               precision    recall  f1-score   support

           0       0.87      0.90      0.88      4549
           1       0.65      0.58      0.61      1454

    accuracy                           0.82      6003
   macro avg       0.76      0.74      0.75      6003
weighted avg       0.82      0.82      0.82      6003
```

Note that because class `1` is defined as the "positive" label that we want to predict (income above 50K), these metrics should be highlighted:

```
Precision: 0.6476043276661515
Recall: 0.5763411279229711
F1: 0.6098981077147017
```

Refer to [here](./slice_evaluation.yaml) to inspect model performance on data slices for values of the categorical features.

## Ethical Considerations

If the outcome of the predictions are used to inform decisions that affect people's lives (such as deciding on loan applications), there is a serious risk of biased decision-making and unwanted feedback loops.
The data set contains sensitive personal data that should almost certainly not be used to inform financial decisions.

## Caveats and Recommendations

- No strategy was implemented to deal with the data imbalances in the input features and target labels. This is an obvious next step to explore.
- The training and evaluation data sets contain sensitive features. Consider modelling approaches that explicitly control for bias on these features.
- The [evaluation results on data slices](./slice_evaluation.yaml) clearly show very poor performance for certain subgroups. This should be investigated further to see if for example collecting more data for these subgroups would be sufficient to counter this discrepancy in model performance.