Introduction:
Hello everyone! Welcome to this exciting tutorial where we’ll learn how to deploy a machine learning model. First, I'll introduce you to my very own machine learning model. Then, we'll dive into how to save the model. Finally, we’ll see how to deploy the model step-by-step.

Introducing the Machine Learning Model:
My model is based on the fascinating Titanic dataset. It predicts whether you would have survived if you were a passenger on the Titanic. All the data and the model are available in my GitHub repository, which you can clone to follow along and create your web application.

Creating the Machine Learning Model:
Let's get started by adding some initial text to our notebook. Make sure you have your notebook ready!

Library Import:
First, we need to import the necessary libraries. This is the foundation of our project.

python code
```
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
```
Regularization:
Regularization is a crucial step, especially since we have a column called "cabin" in our dataset. This helps us handle the data more effectively.

Handling Missing Values:
We often encounter missing values, like in the age column. Here's how we can handle them efficiently:

python code
```
df['Age'].fillna(df['Age'].median(), inplace=True)
```
Processing the Cabin Column:
We’ll split the cabin column into two new columns: one for the cabin class and another for the cabin number. Let's use lambda functions to do this.

python code
```
df['CabinClass'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'Unknown')
df['CabinNumber'] = df['Cabin'].apply(lambda x: ''.join(filter(str.isdigit, str(x))) if pd.notnull(x) else 'Unknown')
```
Handling the Embarked Column:
For the "Embarked" column, we'll use the fillna method to manage missing values:

python code
```
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```
Dropping Unnecessary Columns:
Next, let's drop columns that aren't useful for our model, such as "PassengerId":

python code
```
X = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
```
Creating Pipelines:
We’ll create two pipelines:

StandardScaler for numerical data.
OneHotEncoder for categorical data.
python code
```
numeric_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, ['Age', 'Fare']),
        ('cat', categorical_pipeline, ['Sex', 'Embarked', 'Pclass', 'CabinClass'])
    ])
```
Splitting the Data:
Now, let's split the dataset into training and testing sets:

python code
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
```
Building the Final Pipeline:
We’ll create a final pipeline that first preprocesses the data and then applies a RandomForestClassifier:

python code
```
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

model.fit(X_train, y_train)
```
Model Prediction:
Finally, let's predict the test data and see the accuracy of our model:

python code
```
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy:.2f}')
```
Conclusion:
That’s it! We’ve successfully built, trained, and tested our machine learning model. Now, you know how to deploy it as well. Make sure to check out the GitHub repository for the complete code and dataset. Happy coding!

