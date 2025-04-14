import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols




df = pd.read_csv('Sheet1(Nike_Dataset)_Sheet1.csv')
#check for duplicated rows
duplicated_rows = df.duplicated()
dup_num = duplicated_rows.sum()
if dup_num>0:
     print('The duplicated rows are the following:\n')
     print(df[duplicated_rows])
     print('The total number of duplicated records is: ',dup_num)
else:
     print('No duplicated records were found in the dataset.')

#check for missing value and basic data explore
pd.set_option('display.max_columns', None)
print(df.columns,'\n')
print(df.isna().sum(), '\n')
print(df.describe(), '\n')
print(df.dtypes)
print(df.head())

categorical_columns = df.select_dtypes(include=['object']).columns #check all the object datatype have correct input
for col in categorical_columns:
    print(f"Unique values in '{col}':\n", df[col].unique(), "\n")
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.str.strip().str.lower())

#std of Total_Sales is too high, some value of total_sales might be wrong
df['Computed Sales'] = df['Price per Unit'] * df['Units Sold']
inconsistent_sales = df[df['Computed Sales'] != df['Total Sales']]
print(inconsistent_sales)
print(df.describe())
print(df.dtypes)
df.loc[df['Total Sales'] != df['Computed Sales'], 'Total Sales'] = df['Computed Sales']
print((df['Computed Sales'] != df['Total Sales']).sum())
remaining_issues = df[df['Total Sales'] != df['Computed Sales']]
if remaining_issues.empty:
    print("All Total Sales values have been correctly updated.")
else:
    print(f"There are still {len(remaining_issues)} inconsistent records remaining.")


#change Invoice date formate from object to datetime
df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], format='%d-%m-%Y')
print(df.dtypes)

# Outlier detection: Checking min and max values
print("Min and Max values for key numerical columns:")
print(df[['Price per Unit', 'Total Sales', 'Units Sold','Computed Sales']].agg(['min', 'max']))

# Identify transactions with zero sales
zero_sales = df[df['Total Sales'] == 0]
print(f"Number of transactions with zero sales: {zero_sales.shape[0]}")


# Explore data analysis
# Correlation Matrix
plt.figure(figsize=(8, 6))
corr = df[["Price per Unit", "Total Sales", "Units Sold"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True)
plt.title("Correlation Matrix")
plt.show()

# Sales Trend Over Time
plt.figure(figsize=(12, 6))
df.groupby("Invoice Date")["Total Sales"].sum().plot(kind="line", color="blue", linewidth=2)
plt.title("Total Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Sales Distribution by Method
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=df["Sales Method"].value_counts().index, y=df["Sales Method"].value_counts().values, palette="Blues_r")
plt.title("Sales Distribution by Method")
plt.xlabel("Sales Method")
plt.ylabel("Count")

# Add data labels
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

plt.show()

# MODEL SELECTION#

# Encode categorical variables
df = pd.get_dummies(df, columns=["Region", "Retailer", "Sales Method", "State", 'Product'], drop_first=True)

# Convert all columns to numeric (force non-numeric values to NaN)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN values
df.dropna(inplace=True)

# Define independent variables and target
X = df[["Price per Unit", "Units Sold"]]
y = df["Total Sales"]

# Add constant term for regression
X = sm.add_constant(X)

# Fit Linear Regression Model
model = sm.OLS(y, X).fit()

# Display regression results
print("\nRegression Model Summary:\n", model.summary())



coef_df = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.params,
    "P-Value": model.pvalues
})
coef_df.to_csv("Nike_Regression_Coefficients.csv", index=False)

# # Feature Importance Plot
coef_df = pd.DataFrame({"Variable": X.columns, "Coefficient": model.params})
plt.figure(figsize=(6, 4))
sns.barplot(x="Coefficient", y="Variable", data=coef_df, palette="coolwarm")
plt.title("Feature Importance (Regression Coefficients)")
plt.show()

# Predicted vs. Actual Sales Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=model.predict(X), y=y, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", color="red")  # Reference line
plt.xlabel("Predicted Sales")
plt.ylabel("Actual Sales")
plt.title(f"Predicted vs. Actual Sales (R² = {model.rsquared:.2f})")
plt.grid(True)
plt.show()


# Create a new column to identify zero-sales transactions
df["Zero_Sales"] = df["Total Sales"].apply(lambda x: 1 if x == 0 else 0)

# Display the first few rows to confirm the changes
print(df[["Invoice Date", "Product", "Total Sales", "Zero_Sales"]].head())

# Ensure correct data types
df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])  # Convert date column

# Define independent variables (features) and target variable (sales)
X = df[["Price per Unit", "Units Sold"]]  # Features for predicting sales
y = df["Total Sales"]  # Actual sales

# Add a constant term for regression (intercept)
X = sm.add_constant(X)

# Fit the Linear Regression Model
model = sm.OLS(y, X).fit()

# Create new columns: Predicted Sales and Actual Sales
df["Predicted Sales"] = model.predict(X)  # Model-generated sales predictions
df["Actual Sales"] = y  # Assign actual sales values from dataset

# Display model summary (optional)
print(model.summary())

# Display the first few rows of the updated dataset
print(df[["Invoice Date", "Product", "Total Sales", "Predicted Sales", "Actual Sales"]].head())

save_file = 'clean_Nikegroup2.csv'
df.to_csv(save_file)

print(df.describe(), '\n')