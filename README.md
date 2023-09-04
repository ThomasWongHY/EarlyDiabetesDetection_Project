# EarlyDiabetesDetection_Project

## Data Preprocessing
```python
df = df.replace('No', 0)
df = df.replace('Yes', 1)
df = df.replace('Positive', 1)
df = df.replace('Negative', 0)
df = df.replace('Male', 1)
df = df.replace('Female', 0)

replace = {'Gender':'IsMale'}

df = df.rename(columns = replace)
df.columns.str.lower()

# lowercase everything in columns
df.columns = df.columns.str.lower()
```

## Data Visualization
```python
columns = df.columns[1:]
for column in columns:
    sns.countplot(x=df[column])
    plt.title(column)
    sns.despine()
    plt.show()
```
![image](https://github.com/ThomasWongHY/EarlyDiabetesDetection_Project/assets/86035047/4addaf69-1ef7-4ee6-bad3-0b2cce745c4a)
![image](https://github.com/ThomasWongHY/EarlyDiabetesDetection_Project/assets/86035047/a11620bf-829f-41e4-bcbd-22585cf2b418)

```python
# Check relationship between age and diabetic status
sns.boxplot(x=df['class'], y=df['age'])
```
![image](https://github.com/ThomasWongHY/EarlyDiabetesDetection_Project/assets/86035047/231352c7-243e-4974-bf10-bbd1aacb6521)

```python
# Correlation plot
sns.heatmap(df.corr())
```
![image](https://github.com/ThomasWongHY/EarlyDiabetesDetection_Project/assets/86035047/277828a5-579d-4658-859a-a7550f7c518f)

## Model Building
```python
# prepare our independent and dependent variables
X = df.drop('class', axis=1)
y = df['class']

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y)
```

```python
# Create a Dummy classifier
dummy = DummyClassifier()
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
confusion_matrix(y_test, dummy_pred)
# Check the confusion matrix
confusion_matrix(y_test, dummy_pred)

# Create a Logistic Regression classifier
logr = LogisticRegression(max_iter=10000)
logr.fit(X_train, y_train)
logr_pred = logr.predict(X_test)
# Check the confusion matrix
confusion_matrix(y_test, logr_pred)

# Create a Decision Tree classifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
# Check the confusion matrix
confusion_matrix(y_test, tree_pred)

# Create a Random Forest classifier
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
forest_pred = forest.predict(X_test)
# Check the confusion matrix
confusion_matrix(y_test, forest_pred)
```

## Check the importance of the features
```python
pd.DataFrame({'feature': X.columns, 'importance': forest.feature_importances_}).sort_values('importance', ascending=False)
```
