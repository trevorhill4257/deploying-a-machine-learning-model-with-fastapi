# Model Card

For additional information on model cards see the
[Model Card paper](https://arxiv.org/pdf/1810.03993.pdf).

> NOTE: The metrics below were produced against a SYNTHETIC census.csv that
> shares the schema of the UCI Census Income (Adult) dataset, so the pipeline
> could be run end-to-end in the workspace. Re-run `python train_model.py`
> with the real dataset and update the numbers before final submission.

## Model Details

The model is a scikit-learn `RandomForestClassifier` trained with a fixed
`random_state=42` for reproducibility, using the library's default
hyperparameters. It was built as the capstone for the "Deploying a Machine
Learning Model with FastAPI" project and predicts whether an individual's
annual income exceeds $50K based on U.S. Census demographic attributes.

## Intended Use

The model is intended to demonstrate an end-to-end ML pipeline: training,
inference, slice-based performance monitoring, and deployment behind a REST
API. It is appropriate for educational and demonstration purposes. It should
not be used to make real decisions about individuals, such as lending,
hiring, or eligibility determinations.

## Training Data

The training data is the Census Income dataset (also known as the Adult
dataset). The data contains both continuous attributes (such as age,
hours-per-week, and capital-gain) and categorical attributes (such as
workclass, education, occupation, and native-country). The dataset was split
into 80% training and 20% test using `train_test_split` with
`random_state=42`. Categorical features were one-hot encoded with a
`OneHotEncoder` and the target label was binarized with a `LabelBinarizer`;
both transformers were fit on the training data only.

## Evaluation Data

The evaluation data is the 20% held-out test split, processed with the same
encoder and label binarizer fitted on the training split (using
`training=False`) so that there is no leakage between training and evaluation.

## Metrics

The model was evaluated using precision, recall, and F1 (F-beta with
beta=1). On the held-out test set the model achieved:

- Precision: 0.8456
- Recall: 0.7325
- F1: 0.7850

Per-slice performance for every unique value of each categorical feature was
also computed and written to `slice_output.txt`, which reports the precision,
recall, F1, and row count for each slice.

## Ethical Considerations

The Census Income data encodes sensitive attributes including race, sex, and
native-country, and historical societal biases can be reflected in the data
and learned by the model. The slice-based metrics in `slice_output.txt` are
intended to surface performance disparities across these groups. The model
should not be deployed in any setting that affects individuals without a
thorough fairness assessment.

## Caveats and Recommendations

The model uses default Random Forest hyperparameters and was not tuned, so
performance could likely be improved with hyperparameter optimization and
cross-validation. The dataset reflects a specific historical population and
may not generalize to other populations or time periods. Recommended next
steps include hyperparameter tuning, calibration, fairness auditing across
the data slices, and validation on the real dataset before any practical use.
