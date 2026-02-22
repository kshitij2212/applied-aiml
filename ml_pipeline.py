import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

class NoShowPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.threshold = 0.35
        self.model_name = None

    def preprocess(self,df):
        df = df.copy()

        df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'],utc=True)
        df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'],utc=True)
        df['LeadTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days.clip(lower=0)

        df['DayOfWeek'] = df['AppointmentDay'].dt.dayofweek      

        df['Gender_Male'] = (df['Gender'] == 'M').astype(int)
        if 'No-show' in df.columns:
            df['NoShowBinary'] = (df['No-show'] == 'Yes').astype(int)
        
        df['Age'] = df['Age'].clip(0, 115)

        df['AgeGroup'] = pd.cut(df['Age'],bins=[-1, 12, 18, 35, 55, 75, 120],labels=False).astype(float).fillna(2)

        return df
    

    def train(self, df, model_name, test_size = 0.2 ,threshold = 0.35):
        df = self.preprocess(df)
        self.model_name = model_name
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.feature_cols = ['AgeGroup', 'Gender_Male', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'LeadTime', 'DayOfWeek']

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.feature_cols]
        if 'NoShowBinary' not in df.columns:
            raise ValueError("Target column missing.")
        y = df['NoShowBinary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        

        if model_name == 'Logistic Regression':
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            self.model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

        elif model_name == 'Decision Tree':
            self.model = DecisionTreeClassifier(random_state=42, class_weight='balanced')

        elif model_name == 'Random Forest':
            self.model = RandomForestClassifier(random_state=42, class_weight='balanced')

        else:
            raise ValueError("Invalid model_name")

        self.model.fit(X_train, y_train)

        y_proba = self.model.predict_proba(X_test)[:,1]
        y_pred = (y_proba >= threshold).astype(int)

        if hasattr(self.model, "feature_importances_"):
            importance = dict(zip(self.feature_cols, self.model.feature_importances_))
        elif hasattr(self.model, "coef_"):
            importance = dict(zip(self.feature_cols, self.model.coef_[0]))
        else:
            importance = None

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'feature_importance': importance}


    def predict(self, patient_data):
        if self.model is None or self.feature_cols is None:
            raise ValueError("Model is not trained yet.")

        df = pd.DataFrame([patient_data])
        df = self.preprocess(df)

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[self.feature_cols]

        if self.model_name == 'Logistic Regression':
            X = self.scaler.transform(X)

        prob = self.model.predict_proba(X)[0][1]
        return {
            'probability': float(prob),
            'prediction': int(prob >= self.threshold),
            'threshold_used': self.threshold
        }