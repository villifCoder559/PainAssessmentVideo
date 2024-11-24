from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, GroupKFold,GroupShuffleSplit
from sklearn.metrics import mean_absolute_error as mea
import numpy as np
import matplotlib.pyplot as plt


# class head:
#   def __init__(self,head):
#     if head == 'SVR':
#       self.head = SVR_head

class SVR_head:
  def __init__(self, svr_params):
    self.svr = SVR(**svr_params)

  def plot_mea_per_class(self,y_gt, y_pred, classes):
    """ Plot Mean Absolute Error per class. """
    mae_per_class = []
    for cls in classes:
      idx = np.where(y_gt == cls)
      mae_per_class.append(mea(y_gt[idx], y_pred[idx]))

    plt.figure(figsize=(10, 5))
    plt.bar(classes, mae_per_class, color='blue',width=0.4)
    plt.xlabel('Class')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(classes)  # Show each element in x-axis
    plt.title('Mean Absolute Error per Class')
    plt.show()

  def plot_mea_per_participant(self, y_true, y_pred, subject_ids):
    """ Plot Mean Absolute Error per participant. """
    mae_per_participant = []
    print('subject_ids', subject_ids)
    unique_subjects = np.unique(subject_ids)
    for sub in unique_subjects:
      idx = subject_ids == sub
      mae_per_participant.append(mea(y_true[idx], y_pred[idx]))

    plt.figure(figsize=(10, 5))
    plt.bar(unique_subjects, mae_per_participant, color='green')
    plt.xlabel('Participant')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error per Participant')
    plt.xticks(unique_subjects)  # Show each element in x-axis
    plt.show()

  def fit(self, X_train, y_train, subject_ids):
    """ Evaluation training of SVR model. """

    regressor = self.svr.fit(X_train, y_train)
    y_pred = regressor.predict(X_train)
    # Model evaluation
    # y_pred = regressor.predict(X_test)
    print("Mean absolute Error:", mea(y_train, y_pred))
    print("diffenece:", np.sum(np.abs(y_train - y_pred) >= 1))
    self.plot_mea_per_class(y_train, y_pred, np.unique(y_train))
    self.plot_mea_per_participant(y_train, y_pred, subject_ids)



  def predict(self, X):
    predictions = self.svr.predict(X)
    # rounded_predictions = np.round(predictions).astype(int)
    # print(f'Predicition: {rounded_predictions}')
    return predictions

  def k_fold_cross_validation(self, X, y, groups, k=3):
    """ k-fold cross-validation training of SVR model. """
    # Use dictionary so you cann add w/o changing code
    print('X.shape', X.shape)
    print('y.shapey', y.shape)
    gss = GroupShuffleSplit(n_splits = k)
    results = cross_validate(self.svr, X, y, cv=gss,scoring='neg_mean_absolute_error', groups=groups, return_train_score=True, return_estimator=True)
    # scores = - scores
    # Print the scores for each fold and the mean score
    print("Keys:", results.keys())
    print("Train accuracy:", results['train_score'])
    print("Test accuracy:", results['test_score'])
    # print("Mean test accuracy:", results['test_accuracy'].mean())
    list_split_indices=[]
    for fold, (train_idx, test_idx) in enumerate(gss.split(X, y, groups=groups), 1):
      list_split_indices.append((train_idx,test_idx))
    return list_split_indices,results

  def run_grid_search(self,param_grid, X, y, groups ,k_cross_validation):
    # Initialize GridSearchCV
    gss = GroupShuffleSplit(n_splits=k_cross_validation)
    fig, ax = plt.subplots()
    self._plot_cv_indices(gss, X, y, groups, ax, k_cross_validation)
    plt.tight_layout()
    plt.show()
    grid_search = GridSearchCV(estimator=self.svr, param_grid=param_grid, cv=gss,scoring='neg_mean_absolute_error',return_train_score=True)

    # Fit the grid search to your data
    grid_search.fit(X, y, groups=groups)

    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score}")
    list_split_indices=[]

    for fold, (train_idx, test_idx) in enumerate(gss.split(X, y, groups=groups), 1):
      list_split_indices.append((train_idx,test_idx))
    return grid_search, list_split_indices
  
  def _plot_cv_indices(self,cv, X, y, group, ax, n_splits, lw=20):
    """Create a sample plot for indices of a cross-validation object."""
    use_groups = "Group" in type(cv).__name__
    groups = group if use_groups else None
    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm 
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    # bar_spacing = 2.5
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[(n_splits + 2.2), -0.2],
        xlim=[0, 30],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax
