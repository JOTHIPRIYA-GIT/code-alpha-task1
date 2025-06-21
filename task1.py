import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

np.random.seed(0)
data = pd.DataFrame({
    'income': np.random.normal(50000, 15000, 1000),
    'debts': np.random.normal(10000, 5000, 1000),
    'payment_history': np.random.randint(0, 10, 1000),  
    'age': np.random.randint(21, 60, 1000),
    'credit_lines': np.random.randint(1, 10, 1000),
    'defaulted': np.random.randint(0, 2, 1000)  
})


data['debt_income_ratio'] = data['debts'] / data['income']
features = ['income', 'debts', 'payment_history', 'age', 'credit_lines', 'debt_income_ratio']
X = data[features]
y = data['defaulted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n🔹 {name} Results:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

