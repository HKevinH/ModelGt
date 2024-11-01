import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_gb_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model

def save_model(model, label_encoder, encoder, model_path='gb_model.pkl', encoder_path='onehot_encoder.pkl', label_path='label_encoder.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(label_encoder, label_path)
