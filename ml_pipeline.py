import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

class NoShowPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
    
    def preprocess(self,df):
        df = df.copy()

        df['ScheduledDay'] = pd.to_datetime(df['ScheduleDay'],utc=True)
        df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'],utc=True)
        df['LeadTime'] = (df['AppointmentDay'] - df['ScheduleDay']).dt.days.clip(lower=0)

        df['DayOfWeek'] = df['AppointmentDay'].dt.dayOfWeek
        df['AppointmentHour'] = df['AppointmentDay'].dt.hour
        df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)       

        df['Gender_Male'] = (df['Gender'] == 'M').astype(int)
        df['NoShowBinary'] = (df['No-show'] == 'Yes').astype(int)
        
        df['Age'] = df['Age'].clip(lower=0)
        df['Age'] = df['Age'].clip(upper=115)

        df['AgeGroup'] = pd.cut(df['Age'],bins=[0, 12, 18, 35, 55, 75, 120],labels=[0, 1, 2, 3, 4, 5]).astype(float).fillna(2)

        return df