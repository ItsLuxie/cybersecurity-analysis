import numpy as np 
import pandas as pd
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
    # for filename in filenames:
        # print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')
df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')
df.head()
df.tail()
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
df.dtypes
df.shape
df.columns
sns.set_theme(style="darkgrid", palette="deep")

plt.figure(figsize=(14, 8))


plt.subplot(2, 3, 1)
sns.countplot(data=df, x="Year", order=sorted(df["Year"].unique()), color="darkblue")
plt.xticks(rotation=45)
plt.title("Cyber Attacks Over the Years")

plt.subplot(2, 3, 2)
top_countries = df["Country"].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index, color="darkred")
plt.title("Top 10 Affected Countries")

plt.subplot(2, 3, 3)
df["Attack Type"].value_counts().plot.pie(autopct="%1.1f%%", colors=sns.color_palette("pastel"))
plt.ylabel("")
plt.title("Attack Types Distribution")

plt.subplot(2, 3, 4)
top_industries = df["Target Industry"].value_counts().head(10)
sns.barplot(y=top_industries.index, x=top_industries.values, color="darkgreen")
plt.title("Most Targeted Industries")

plt.subplot(2, 3, 5)
sns.boxplot(data=df, y="Financial Loss (in Million $)", color="purple")
plt.yscale("log")
plt.title("Financial Loss Distribution")

plt.subplot(2, 3, 6)
sns.kdeplot(df["Incident Resolution Time (in Hours)"], fill=True, color="orange")
plt.title("Incident Resolution Time Distribution")

plt.tight_layout()
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# print(df.info())
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
# Encoding Categorical Features
categorical_columns = ['Country', 'Year', 'Attack Type', 'Target Industry', 
                       'Attack Source', 'Security Vulnerability Type', 'Defense Mechanism Used']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  
    
X = df.drop(columns=["Attack Type"])  
y = df["Attack Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Store results
results = {}
# Machine Learning Models
models = {
    "Decision Tree": DecisionTreeClassifier(),
}
# Training and Evaluating Models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    results[name] = acc

# Display Model Performance
print("\nModel Performance Summary:")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}") 