import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data
def load_and_preprocess_data():
    local_file_path = 'car.data'  
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv(local_file_path, names=columns)
    df = df.drop_duplicates()
    df['buying'] = df['buying'].replace({'vhigh': 3, 'high': 2, 'med': 1, 'low': 0})
    df['maint'] = df['maint'].replace({'vhigh': 3, 'high': 2, 'med': 1, 'low': 0})
    df['doors'] = df['doors'].replace({'2': 0, '3': 1, '4': 2, '5more': 3})
    df['persons'] = df['persons'].replace({'2': 0, '4': 1, 'more': 2})
    df['lug_boot'] = df['lug_boot'].replace({'small': 0, 'med': 1, 'big': 2})
    df['safety'] = df['safety'].replace({'low': 0, 'med': 1, 'high': 2})
    df['class'] = df['class'].replace({'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3})
    return df

# Train and save models
def train_and_save_models():
    df = load_and_preprocess_data()
    X = df.drop(columns=['doors', 'class'])
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'rf_model.pkl')
    rf_pred = rf_model.predict(X_test)

    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
    print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

    # Hyperparameter tuning
    rf_param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 6]
    }

    rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='accuracy')
    rf_grid_search.fit(X_train, y_train)

    print("Best Random Forest Params: ", rf_grid_search.best_params_)
    print("Best Random Forest Score: ", rf_grid_search.best_score_)

# Convert label to category name
def label_to_category(label):
    categories = {0: 'unacc', 1: 'acc', 2: 'good', 3: 'vgood'}
    return categories.get(label, 'Unknown')

# Prediction
def predict(input_data):
    model_filename = 'rf_model.pkl'
    model = joblib.load(model_filename)

    # Secretly drop the 'doors' feature
    features = [
        int(input_data['buying']),
        int(input_data['maint']),
        int(input_data['persons']),
        int(input_data['lug_boot']),
        int(input_data['safety'])
    ]

    features = np.array(features).reshape(1, -1)  # Reshape for a single sample
    prediction = model.predict(features)
    return label_to_category(prediction[0])

if __name__ == '__main__':
    train_and_save_models()
