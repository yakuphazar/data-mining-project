# Required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadstat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Reading the dataset
file_path = '/Users/bmac/Documents/PycharmProjects/Datamining/6.sav'
CreditRiskData, meta = pyreadstat.read_sav(file_path)
#print(CreditRiskData.columns.tolist())

## 2. EDA:

# print("Dataset Information:")
# print(CreditRiskData.info())
#
# #Basic statistics for numeric columns
# print("\nNumerical Features Description:")
# print(CreditRiskData.describe())
#
# # Check for missing values
# print("\nMissing Values:")
# print(CreditRiskData.isnull().sum())
#
# #See unique values
# print("\nUnique Values in Categorical Features:")
# for column in CreditRiskData.columns:
#     print(f"{column}: {CreditRiskData[column].nunique()} unique values")

#Predict the outliers
#print("\nCredit Amount Statistics:")
#print(CreditRiskData['camt'].describe())

# # Identifying and transforming numeric columns
# numerical_columns = ['camt', 'duration', 'age', 'numcred', 'numliab', 'instrate']
#
# # Calculating the number of outliers in each column
# outlier_counts = {}
# for column in numerical_columns:
#     # Analyze column by converting it to numeric type
#     CreditRiskData[column] = pd.to_numeric(CreditRiskData[column], errors='coerce')
#     outlier_counts[column] = detect_outliers(CreditRiskData, column)
#
# # Print results
# print("Outlier Counts by Column:")
# for column, count in outlier_counts.items():
#     print(f"{column}: {count} outliers")

# #Age Disturbution
# sns.histplot(CreditRiskData['age'], bins=20, kde=True, color='blue')
# plt.title("Age Distribution")
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.show()


# # Credit Amount Distribution
# sns.boxplot(CreditRiskData['camt'], color='orange')
# plt.title("Credit Amount Distribution")
# plt.xlabel("Credit Amount")
# plt.show()

# # Age vs Housing by Risk
# sns.violinplot(x="housng", y="age", data=CreditRiskData, hue="chist", split=True, palette="muted")
# plt.title("Age vs Housing by Account Status")
# plt.xlabel("Housing")
# plt.ylabel("Age")
# plt.show()
#
# # Credit Amount vs Purpose by Account Status
# sns.boxplot(x="reason", y="camt", data=CreditRiskData, hue="chist", palette="Set2")
# plt.title("Credit Amount vs Purpose by Account Status")
# plt.xlabel("Purpose")
# plt.ylabel("Credit Amount")
# plt.xticks(rotation=45)
# plt.show()

# # Housing vs Risk Crosstab
# ct = pd.crosstab(CreditRiskData['housng'], CreditRiskData['chist'])
# sns.heatmap(ct, annot=True, cmap="coolwarm", fmt="d")
# plt.title("Housing vs Account Status Crosstab")
# plt.show()

# # Good Credit: No debt history (1) ve No current debt (2)
# good_credit = CreditRiskData[CreditRiskData['chist'].isin([1, 2])]['age']
#
# # Bad Credit: Critical account (5)
# bad_credit = CreditRiskData[CreditRiskData['chist'] == 5]['age']
#

# #Number of Existing Credits Disturbution
# plt.figure(figsize=(8, 5))
# CreditRiskData['numcred'].value_counts().sort_index().plot(kind='bar', color='salmon', edgecolor='black')
# plt.title('Frequency of numcred')
# plt.xlabel('Number of Credits')
# plt.ylabel('Frequency')
# plt.grid(axis='y', alpha=0.75)
# plt.show()

## 3. DATA PREPROCESSING:

CreditRiskData.rename(columns={
    'duration': 'CreditDuration',
    'camt': 'CreditAmount',
    'instrate': 'InstallmentRate',
    'residlen': 'ResidenceDuration',
    'age': 'Age',
    'numcred': 'NumberOfCredits',
    'numliab': 'NumberOfLiabilities',
    'chks': 'CheckingAccount',
    'chist': 'CreditHistory',
    'reason': 'CreditPurpose',
    'savngs': 'SavingsAccount',
    'lenemp': 'EmploymentLength',
    'perstat': 'PersonalStatus',
    'othdebt': 'OtherDebtors',
    'prpownr': 'PropertyOwner',
    'othnstal': 'OtherInstallments',
    'housng': 'Housing',
    'emptype': 'EmploymentType',
    'telephne': 'HasTelephone',
    'forworkr': 'ForeignWorker'
}, inplace=True)

# # Before Outlier Treatment
# sns.boxplot(CreditRiskData['CreditAmount'])
# plt.title("Credit Amount Before Outlier Treatment")
# plt.show()

# Define outlier detection thresholds
def outlier_thresholds(dataframe, variable):
    Q1 = dataframe[variable].quantile(0.25)
    Q3 = dataframe[variable].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

# Replace outliers with threshold values
def replace_with_thresholds(dataframe, variable):
    lower_bound, upper_bound = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < lower_bound), variable] = lower_bound
    dataframe.loc[(dataframe[variable] > upper_bound), variable] = upper_bound

# Apply the method to numerical variables
for column in ['CreditAmount', 'CreditDuration', 'Age', 'NumberOfCredits', 'NumberOfLiabilities']:
    replace_with_thresholds(CreditRiskData, column)

##Outlier Amount After
# def has_outliers(dataframe, variable):
#     Q1 = dataframe[variable].quantile(0.25)
#     Q3 = dataframe[variable].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = dataframe[(dataframe[variable] < lower_bound) | (dataframe[variable] > upper_bound)]
#     return outliers.shape[0]
#
# outlier_summary = {}
# for column in ['CreditAmount', 'CreditDuration', 'Age', 'NumberOfCredits', 'NumberOfLiabilities']:
#     outlier_summary[column] = has_outliers(CreditRiskData, column)
#
# print("Outlier Counts by Column:")
# print(outlier_summary)

# # After Outlier Treatment
# for column in ['CreditAmount', 'CreditDuration', 'Age', 'NumberOfCredits', 'NumberOfLiabilities']:
#     replace_with_thresholds(CreditRiskData, column)
#
# sns.boxplot(CreditRiskData['CreditAmount'])
# plt.title("Credit Amount After Outlier Treatment")
# plt.show()

# # Before Encoding: Display unique values for all categorical columns
# print("Unique Values in Categorical Columns Before Encoding:")
# categorical_columns = ['Housing', 'CreditPurpose', 'CreditHistory', 'CheckingAccount', 'SavingsAccount',
#                        'EmploymentLength', 'PersonalStatus', 'OtherDebtors', 'OtherInstallments',
#                        'PropertyOwner', 'EmploymentType']
#
# for column in categorical_columns:
#     if column in CreditRiskData.columns:
#         print(f"{column}: {CreditRiskData[column].unique()}")
#     else:
#         print(f"{column}: Not found in the dataset")

# Map nominal variables with specific category names
CreditRiskData['CreditHistory'] = CreditRiskData['CreditHistory'].map({
    1.0: 'NoDebtHistory',  # No previous debt history
    2.0: 'NoCurrentDebt',  # No current debt
    3.0: 'PaymentsCurrent',  # Payments are up-to-date
    4.0: 'PaymentsDelayed',  # Payments are delayed
    5.0: 'CriticalAccount'  # Critical account status
})

CreditRiskData['CreditPurpose'] = CreditRiskData['CreditPurpose'].map({
    1.0: 'NewCar',  # Loan purpose: New car
    2.0: 'UsedCar',  # Loan purpose: Used car
    3.0: 'Furniture',  # Loan purpose: Furniture
    4.0: 'RadioTV',  # Loan purpose: Radio/TV
    5.0: 'Appliances',  # Loan purpose: Appliances
    6.0: 'Repairs',  # Loan purpose: Repairs
    7.0: 'Education',  # Loan purpose: Education
    8.0: 'Vacation',  # Loan purpose: Vacation
    9.0: 'Retraining',  # Loan purpose: Retraining
    10.0: 'Business',  # Loan purpose: Business
    33.0: 'Other'  # Loan purpose: Other
})

CreditRiskData['Housing'] = CreditRiskData['Housing'].map({
    1.0: 'Rent',  # Housing type: Renting
    2.0: 'Own',  # Housing type: Own house
    3.0: 'Free'  # Housing type: Free accommodation
})

CreditRiskData['EmploymentType'] = CreditRiskData['EmploymentType'].map({
    1.0: 'UnskilledNonResident',  # Unskilled and non-resident
    2.0: 'UnskilledResident',  # Unskilled and resident
    3.0: 'SkilledEmployee',  # Skilled employee
    4.0: 'ManagementOrSelfEmployed'  # Management or self-employed
})

CreditRiskData['PropertyOwner'] = CreditRiskData['PropertyOwner'].map({
    1.0: 'RealEstateSavings',  # PropertyOwner type: Real estate or building savings
    2.0: 'BuildingSavings',    # PropertyOwner type: Building savings
    3.0: 'CarOrOther',         # PropertyOwner type: Car or other assets
    4.0: 'UnknownOrNone'       # PropertyOwner type: Unknown or none
})

# Specify nominal columns for one-hot encoding
nominal_columns = ['CreditHistory', 'CreditPurpose', 'Housing', 'EmploymentType', 'PropertyOwner']

# Apply one-hot encoding while preserving the human-readable column names
CreditRiskData_encoded = pd.get_dummies(CreditRiskData, columns=nominal_columns, drop_first=False)

# Convert numeric-like columns to float if possible
for col in CreditRiskData_encoded.select_dtypes(include=['object']).columns:
    try:
        CreditRiskData_encoded[col] = CreditRiskData_encoded[col].astype('float')
    except ValueError:
        print(f"Column {col} could not be converted to float and remains as object.")

# # Display column names after one-hot encoding
# print("Columns After Encoding:")
# print(CreditRiskData_encoded.columns)
#
# # Check Data Types to Ensure No Object Types Remain
# print("\nData Types After Encoding:")
# print(CreditRiskData_encoded.dtypes)

# # Before Normalization
# print("Numerical Features Before Normalization:")
# numerical_columns = CreditRiskData_encoded.select_dtypes(include=['float64']).columns
# print(CreditRiskData_encoded[numerical_columns].describe())

#
# sns.histplot(CreditRiskData_encoded['CreditAmount'], bins=20, kde=True, color='blue')
# plt.title("Credit Amount Before Normalization")
# plt.show()

# Detect and transform Boolean columns
boolean_columns = CreditRiskData_encoded.select_dtypes(include=['bool']).columns
for col in boolean_columns:
    CreditRiskData_encoded[col] = CreditRiskData_encoded[col].astype(float)


numerical_columns = CreditRiskData_encoded.select_dtypes(include=['float64']).columns

# Apply MinMaxScaler
scaler = MinMaxScaler()
CreditRiskData_encoded[numerical_columns] = scaler.fit_transform(CreditRiskData_encoded[numerical_columns])

# # After Normalization
# print("Numerical Features After Normalization:")
# print(CreditRiskData_encoded[numerical_columns].describe())
#
# sns.histplot(CreditRiskData_encoded['CreditAmount'], bins=20, kde=True, color='green')
# plt.title("Credit Amount After Normalization")
# plt.show()

#print("Sample Rows from Dataset:")
#print(CreditRiskData_encoded.head())

#print(CreditRiskData_encoded.dtypes)

#print(CreditRiskData_encoded.columns.tolist())

# # Ensure only numerical columns are selected for correlation
# numerical_columns = CreditRiskData_encoded.select_dtypes(include=['float64']).columns
#
# # Create the correlation matrix using only numerical columns
# correlation_matrix = CreditRiskData_encoded[numerical_columns].corr()
#
# # Fill missing correlations with 0
# correlation_matrix = correlation_matrix.fillna(0)
#
# # Set figure size and properties
# plt.figure(figsize=(30, 25))  # Much larger figure size for better spacing
# sns.heatmap(
#     correlation_matrix,
#     annot=True,             # Display the correlation values in each cell
#     fmt=".2f",              # Format the numbers to show two decimal places
#     cmap="coolwarm",        # Use the "coolwarm" color scale for the heatmap
#     cbar=True,              # Include a color bar on the side
#     linewidths=2,           # Increase line width for better readability
#     square=True,            # Make the cells square-shaped
#     annot_kws={"size": 10}  # Slightly decrease font size of the numbers inside the heatmap
# )
#
# # Customize X and Y axis labels
# plt.xticks(rotation=45, ha="right", fontsize=14)  # Rotate X-axis labels for better visibility and set font size
# plt.yticks(fontsize=14)                           # Set font size for Y-axis labels
# plt.title("Correlation Heatmap", fontsize=24)    # Add a title to the heatmap with a larger font size
# plt.tight_layout()  # Adjust the layout to prevent overlap of elements
# plt.show()


# 4.MODELLING

#Defining Credit Risk Colunmn
#Calculate the averages
median_amount = CreditRiskData_encoded['CreditAmount'].median()
median_duration = CreditRiskData_encoded['CreditDuration'].median()

# Create CreditRisk column
CreditRiskData_encoded['CreditRisk'] = CreditRiskData_encoded.apply(
    lambda row: 0 if (row['CreditAmount'] > median_amount or
                      row['CreditDuration'] > median_duration or
                      row['CreditHistory_PaymentsDelayed'] == 1 or
                      row['CreditHistory_CriticalAccount'] == 1)
    else 1,
    axis=1
)

# #See median values
# median_amount = CreditRiskData_encoded['CreditAmount'].median()
# median_duration = CreditRiskData_encoded['CreditDuration'].median()
#
# print(f"Median CreditAmount: {median_amount}")
# print(f"Median CreditDuration: {median_duration}")
#
# #See Good Credit Risk Accounts
# good_count = CreditRiskData_encoded['CreditRisk'].value_counts()[1]
# print(f"Number of good (1) credits: {good_count}")
#
# #See Bad Credit Risk Accounts
# bad_count = CreditRiskData_encoded['CreditRisk'].value_counts()[0]
# print(f"Number of bad (0) credits: {bad_count}")

#Test Split

X = CreditRiskData_encoded.drop(columns=['CreditRisk'])
y = CreditRiskData_encoded['CreditRisk']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# print(f"Train set: {X_train.shape}, Validation set: {X_valid.shape}, Test set: {X_test.shape}")
# print("Class distribution in Train Set:", y_train.value_counts())
# print("Class distribution in Validation Set:", y_valid.value_counts())
# print("Class distribution in Test Set:", y_test.value_counts())

# Repartition by removing columns that have a strong relationship with the target
X_filtered = X.drop(columns=['CreditAmount', 'CreditDuration'])

# Repartitioning the data
X_train, X_temp, y_train, y_temp = train_test_split(
    X_filtered, y, test_size=0.3, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# # Print split summary
# print(f"Train set: {X_train.shape}, Validation set: {X_valid.shape}, Test set: {X_test.shape}")
# print("Class distribution in Train Set:\n", y_train.value_counts())
# print("Class distribution in Validation Set:\n", y_valid.value_counts())
# print("Class distribution in Test Set:\n", y_test.value_counts())
#
# # Combine class distributions
# split_labels = ['Train', 'Validation', 'Test']
# class_counts = [
#     y_train.value_counts().sort_index(),
#     y_valid.value_counts().sort_index(),
#     y_test.value_counts().sort_index()
# ]
# class_0 = [class_counts[i][0] for i in range(3)]
# class_1 = [class_counts[i][1] for i in range(3)]
#
# # Plotting
# x = range(len(split_labels))
# width = 0.35
#
# plt.bar(x, class_0, width, label='Bad Credit Risk (0)', color='red')
# plt.bar([p + width for p in x], class_1, width, label='Good Credit Risk (1)', color='green')
#
# plt.xlabel('Dataset Split')
# plt.ylabel('Number of Instances')
# plt.title('Class Distribution Across Splits')
# plt.xticks([p + width / 2 for p in x], split_labels)
# plt.legend()
# plt.show()

# Random Forest Model
rfc = RandomForestClassifier(max_depth=5, n_estimators=50, random_state=42)
rfc.fit(X_train, y_train)

# Performance Evaluation
y_pred_valid = rfc.predict(X_valid)
# print("Validation Performance (Random Forest):")
# print(classification_report(y_valid, y_pred_valid))
# print("Confusion Matrix (Validation):")
# print(confusion_matrix(y_valid, y_pred_valid))

# # Features Importances
# feature_importances_rfc = pd.Series(rfc.feature_importances_, index=X_train.columns)
#
# # Top 10
# plt.figure(figsize=(10, 6))
# feature_importances_rfc.nlargest(10).plot(kind='barh', color='green')
# plt.title("Top 10 Feature Importances (Random Forest)", fontsize=14)
# plt.xlabel("Importance Score", fontsize=12)
# plt.ylabel("Features", fontsize=12)
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# # Random Forest Cross-Validation and Test Accuracy
# cv_scores_rfc = cross_val_score(rfc, X_train, y_train, cv=10, scoring='accuracy')
#
# y_test_pred_rfc = rfc.predict(X_test)
# accuracy_test_rfc = accuracy_score(y_test, y_test_pred_rfc)
#
# print("Accuracy of the Random Forest model on Testing Sample Data:", round(accuracy_test_rfc, 2))
# print("\nAccuracy values for 10-fold Cross Validation:\n", cv_scores_rfc)
# print("\nFinal Average Accuracy of the Random Forest model:", round(cv_scores_rfc.mean(), 2))


# Decision Tree Classifier
dtc = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)
dtc.fit(X_train, y_train)

# # Performance Evaluation
y_pred_valid_dtc = dtc.predict(X_valid)
# print("Validation Performance (Decision Tree):")
# print(classification_report(y_valid, y_pred_valid_dtc))
# print("Confusion Matrix (Validation):")
# print(confusion_matrix(y_valid, y_pred_valid_dtc))

# # Decision Tree Feature Importance
# feature_importances_dtc = pd.Series(dtc.feature_importances_, index=X_train.columns)
#
# # Top 10
# plt.figure(figsize=(10, 6))
# feature_importances_dtc.nlargest(10).plot(kind='barh', color='blue')
# plt.title("Top 10 Feature Importances (Decision Tree)", fontsize=14)
# plt.xlabel("Importance Score", fontsize=12)
# plt.ylabel("Features", fontsize=12)
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# # Exporting the tree to Graphviz format
# dot_data = export_graphviz(
#     dtc,                              # Trained Decision Tree Classifier
#     out_file=None,                    # Export as a string
#     feature_names=X_train.columns,    # Feature names
#     class_names=['Bad Risk', 'Good Risk'],  # Target classes
#     filled=True,                      # Color the nodes
#     rounded=True,                     # Rounded edges
#     special_characters=True           # Use special characters
# )
#
# # Visualize the tree using Graphviz
# graph = graphviz.Source(dot_data)
# graph.format = "png"  # Save as PNG if needed
# graph.render("decision_tree_visualization")  # Saves the file to the current directory
# graph.view()  # Opens the rendered file

# # Perform 10-Fold Cross Validation
# cv_scores_dtc = cross_val_score(dtc, X_train, y_train, cv=10, scoring='accuracy')
#
# # Test Performance
# y_test_pred = dtc.predict(X_test)
# accuracy_test = accuracy_score(y_test, y_test_pred)
#
# # Print Cross-Validation Scores and Test Accuracy
# print("Accuracy of the model on Testing Sample Data:", round(accuracy_test, 2))
# print("\nAccuracy values for 10-fold Cross Validation:\n", cv_scores_dtc)
# print("\nFinal Average Accuracy of the model:", round(cv_scores_dtc.mean(), 2))

# # Extract a single tree from the Random Forest
# single_tree = rfc.estimators_[4]  # Select the 5th tree (index starts at 0)
#
# # Export the tree to Graphviz format
# dot_data = export_graphviz(
#     single_tree,
#     out_file=None,
#     feature_names=X_train.columns,  # Feature names
#     class_names=['Bad Risk', 'Good Risk'],  # Target class names
#     filled=True,  # Color nodes
#     rounded=True,  # Rounded edges
#     special_characters=True  # Use special characters
# )
#
# # Visualize the single tree
# graph = graphviz.Source(dot_data)
# graph.format = "png"  # Save as PNG if needed
# graph.render("random_forest_single_tree_visualization")  # Saves the file
# graph.view()  # Opens the rendered file

# # Logistic Regression Model
# lr = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
# lr.fit(X_train, y_train)
#
# # Predictions on Validation and Test Sets
# y_pred_valid = lr.predict(X_valid)
# y_pred_test = lr.predict(X_test)
#
# # Validation Performance
# print("Validation Performance (Logistic Regression):")
# print(classification_report(y_valid, y_pred_valid))
# print("Confusion Matrix (Validation):")
# print(confusion_matrix(y_valid, y_pred_valid))
#
# # Test Performance
# print("\nTest Performance (Logistic Regression):")
# print(classification_report(y_test, y_pred_test))
# print("Confusion Matrix (Test):")
# print(confusion_matrix(y_test, y_pred_test))
#
# # Performance on Test Set
# from sklearn.metrics import accuracy_score, f1_score
#
# test_accuracy = accuracy_score(y_test, y_pred_test)
# f1_test = f1_score(y_test, y_pred_test, average='weighted')
# print(f"\nAccuracy of the model on Testing Sample Data: {test_accuracy:.2f}")
#
# # Cross-Validation Scores
# cv_scores = cross_val_score(lr, X_train, y_train, cv=10, scoring='f1_weighted')
# print("\nAccuracy values for 10-fold Cross Validation:\n", cv_scores)
# print("\nFinal Average Accuracy of the model:", round(cv_scores.mean(), 2))
#
# # ROC Curve and AUC
# y_probs_valid = lr.predict_proba(X_valid)[:, 1]  # Probability estimates for the positive class
# fpr, tpr, thresholds = roc_curve(y_valid, y_probs_valid)
# auc = roc_auc_score(y_valid, y_probs_valid)
#
# # Plotting the ROC Curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.2f})", color='red')
# plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve - Logistic Regression")
# plt.legend(loc="lower right")
# plt.grid(alpha=0.3)
# plt.show()


# # Adaboost Model
# adb = AdaBoostClassifier(n_estimators=100, random_state=42)
# adb.fit(X_train, y_train)
#
# # Validation Set Performance Evaluation
# y_pred_valid_adb = adb.predict(X_valid)
# print("Validation Performance (Adaboost):")
# print(classification_report(y_valid, y_pred_valid_adb))
# print("Confusion Matrix (Validation):")
# print(confusion_matrix(y_valid, y_pred_valid_adb))
#
# # Test Set Performance Evaluation
# y_pred_test_adb = adb.predict(X_test)
# print("Testing Performance (Adaboost):")
# print(classification_report(y_test, y_pred_test_adb))
# print("Confusion Matrix (Testing):")
# print(confusion_matrix(y_test, y_pred_test_adb))
#
# # Cross-validation
# cross_val_scores_adb = cross_val_score(adb, X_train, y_train, cv=10, scoring='accuracy')
#
# # Accuracy on Test Data
# accuracy_test_adb = accuracy_score(y_test, y_pred_test_adb)
#
# # Ek Çıktılar
# print("\nAccuracy of the model on Testing Sample Data: {:.2f}".format(accuracy_test_adb))
# print("\nAccuracy values for 10-fold Cross Validation:\n", cross_val_scores_adb)
# print("\nFinal Average Accuracy of the model: {:.2f}".format(cross_val_scores_adb.mean()))
#
# # Feature Importances
# feature_importances_adb = pd.Series(adb.feature_importances_, index=X_train.columns)
#
# plt.figure(figsize=(12, 8))  # Daha geniş ve uzun bir grafik boyutu
# feature_importances_adb.nlargest(10).plot(kind='barh', color='red')
#
# plt.title("Top 10 Feature Importances (Adaboost)", fontsize=16)
# plt.xlabel("Importance Score", fontsize=14)
# plt.ylabel("Features", fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
#
# # Y ekseni yazılarını tam göstermek için hizalama
# plt.tight_layout()  # Grafiği sıkıştırarak tüm etiketlerin sığmasını sağlar
# plt.show()


# #adaboost decision tree 4th
# plt.figure(figsize=(30, 15))
# plot_tree(adb.estimators_[3],
#           feature_names=X_train.columns,
#           class_names=['Bad Credit Risk', 'Good Credit Risk'],
#           filled=True,
#           rounded=True,
#           fontsize=12)
# plt.title("Adaboost - 4th Decision Tree Visualization", fontsize=16)
# plt.show()
#
#
# # KNN Model
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
#
# # Performance Evaluation
# y_pred_valid_knn = knn.predict(X_valid)
# print("Test Performance (KNN):")
# print(classification_report(y_valid, y_pred_valid_knn))
# print("Confusion Matrix (Validation):")
# print(confusion_matrix(y_valid, y_pred_valid_knn))
#
# # Cross-validation
# cross_val_scores_knn = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
#
# # Performance Evaluation on Test Set
# y_pred_test_knn = knn.predict(X_test)
# accuracy_test_knn = accuracy_score(y_test, y_pred_test_knn)
#
# # Print Results
# print(f"Accuracy of the KNN model on Testing Sample Data: {round(accuracy_test_knn, 2)}")
# print("Accuracy values for 10-fold Cross Validation (KNN):", cross_val_scores_knn)
# print(f"Final Average Accuracy of the KNN model: {round(cross_val_scores_knn.mean(), 2)}")
#
#
# # SVM Model
# svm = SVC(kernel='rbf', probability=True, random_state=42)
# svm.fit(X_train, y_train)
#
# # Performance Evaluation
# y_pred_valid_svm = svm.predict(X_valid)
# print("Test Performance (SVM):")
# print(classification_report(y_valid, y_pred_valid_svm))
# print("Confusion Matrix (Validation):")
# print(confusion_matrix(y_valid, y_pred_valid_svm))
#
# # Cross-validation
# cross_val_scores_svm = cross_val_score(svm, X_train, y_train, cv=10, scoring='accuracy')
#
# # Performance Evaluation on Test Set
# y_pred_test_svm = svm.predict(X_test)
# accuracy_test_svm = accuracy_score(y_test, y_pred_test_svm)
#
# # Print Results
# print(f"Accuracy of the SVM model on Testing Sample Data: {round(accuracy_test_svm, 2)}")
# print("Accuracy values for 10-fold Cross Validation (SVM):", cross_val_scores_svm)
# print(f"Final Average Accuracy of the SVM model: {round(cross_val_scores_svm.mean(), 2)}")
#
#
# # Naive Bayes Model
# nb = GaussianNB()
# nb.fit(X_train, y_train)
#
# # Performance Evaluation
# y_pred_valid_nb = nb.predict(X_valid)
# print("Test Performance (Naive Bayes):")
# print(classification_report(y_valid, y_pred_valid_nb))
# print("Confusion Matrix (Validation):")
# print(confusion_matrix(y_valid, y_pred_valid_nb))
#
# # Cross-validation
# cross_val_scores_nb = cross_val_score(nb, X_train, y_train, cv=10, scoring='accuracy')
#
# # Performance Evaluation on Test Set
# y_pred_test_nb = nb.predict(X_test)
# accuracy_test_nb = accuracy_score(y_test, y_pred_test_nb)
#
# # Print Results
# print(f"Accuracy of the Naive Bayes model on Testing Sample Data: {round(accuracy_test_nb, 2)}")
# print("Accuracy values for 10-fold Cross Validation (Naive Bayes):", cross_val_scores_nb)
# print(f"Final Average Accuracy of the Naive Bayes model: {round(cross_val_scores_nb.mean(), 2)}")
#

# # Performance metrics
# models = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'KNN', 'SVM', 'Naive Bayes', 'AdaBoost']
# accuracy = [0.81, 0.81, 0.81, 0.75, 0.76, 0.65, 0.81]
# f1_score = [0.79, 0.78, 0.77, 0.78, 0.74, 0.66, 0.79]
# precision = [0.87, 0.85, 0.83, 0.83, 0.81, 1.00, 0.85]
# recall = [0.86, 0.92, 0.94, 0.93, 0.88, 0.53, 0.92]
# cv_mean = [0.78, 0.79, 0.77, 0.78, 0.79, 0.66, 0.79]
#
# # DataFrame
# performance_data = pd.DataFrame({
#     'Model': models,
#     'Accuracy': accuracy,
#     'F1-Score': f1_score,
#     'Precision': precision,
#     'Recall': recall,
#     'CV Mean': cv_mean
# })
#
# print(performance_data)
#
# # Line Chart
# plt.figure(figsize=(10, 6))
# plt.bar(models, accuracy, color='skyblue')
# plt.title('Model Accuracy Comparison')
# plt.xlabel('Models')
# plt.ylabel('Accuracy')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Random Forest ve AdaBoost
# features = ["Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8", "Feature9", "Feature10"]
# importance_rf = [0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02]
# importance_adb = [0.20, 0.19, 0.14, 0.13, 0.11, 0.07, 0.06, 0.05, 0.03, 0.02]
#
# # Graph
# x = np.arange(len(features))
#
# plt.figure(figsize=(10, 6))
# plt.barh(x - 0.2, importance_rf, height=0.4, label="Random Forest", color='blue')
# plt.barh(x + 0.2, importance_adb, height=0.4, label="AdaBoost", color='red')
#
# # Y axis
# plt.yticks(x, features)
# plt.xlabel("Importance Score")
# plt.title("Top Features: Random Forest vs AdaBoost")
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # Performance datas
# models = ["Decision Tree", "Random Forest", "Logistic Regression", "KNN", "SVM", "Naive Bayes", "AdaBoost"]
# precision = [0.87, 0.85, 0.83, 0.83, 0.81, 1.00, 0.85]
# recall = [0.86, 0.92, 0.94, 0.93, 0.88, 0.53, 0.92]
# f1_score = [0.86, 0.78, 0.77, 0.78, 0.74, 0.66, 0.79]
# accuracy = [0.81, 0.81, 0.81, 0.75, 0.76, 0.65, 0.81]
#
# # Bar widht x position
# bar_width = 0.2
# x = np.arange(len(models))
#
# # Graph
# plt.figure(figsize=(12, 6))
# plt.bar(x - bar_width*1.5, precision, width=bar_width, label="Precision")
# plt.bar(x - bar_width/2, recall, width=bar_width, label="Recall")
# plt.bar(x + bar_width/2, f1_score, width=bar_width, label="F1-Score")
# plt.bar(x + bar_width*1.5, accuracy, width=bar_width, label="Accuracy")
#
# # X
# plt.xticks(x, models, rotation=45)
# plt.ylabel("Scores")
# plt.title("Model Performance Comparison")
# plt.legend()
# plt.tight_layout()
# plt.show()


