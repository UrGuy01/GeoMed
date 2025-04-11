import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load datasets
print("Loading datasets...")
dataset = pd.read_csv('training_data/Training.csv')
symptom_severity = pd.read_csv('training_data/Symptom-severity.csv')

# Create a dictionary mapping symptoms to their severity
severity_dict = {}
for index, row in symptom_severity.iterrows():
    symptom = row['Symptom'].strip().lower().replace(' ', '_')
    severity = row['weight']
    severity_dict[symptom] = severity

# Fill missing severities with a default value
default_severity = 5
print(f"Found {len(severity_dict)} symptoms with severity information")

# Prepare features and target
X = dataset.drop('prognosis', axis=1)
column_names = X.columns
y = dataset['prognosis']

# Create a weighted feature matrix using severity instead of binary
X_weighted = X.copy()
for symptom in column_names:
    if symptom in severity_dict:
        X_weighted.loc[X_weighted[symptom] == 1, symptom] = severity_dict[symptom]
    else:
        # If the symptom is not in our severity dictionary, use the default severity
        X_weighted.loc[X_weighted[symptom] == 1, symptom] = default_severity

# Encode disease labels
le = LabelEncoder()
le.fit(y)
Y = le.transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_weighted, Y, test_size=0.2, random_state=42)

print("Training improved models...")

# Create and train multiple models with regularization
svc_model = SVC(kernel='linear', C=1.0, probability=True)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Ensemble model (Voting Classifier)
ensemble_model = VotingClassifier(
    estimators=[
        ('svc', svc_model),
        ('rf', rf_model)
    ],
    voting='soft'
)

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Evaluate with cross-validation
cv_scores = cross_val_score(ensemble_model, X_weighted, Y, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Final evaluation on test set
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
with open('improved_ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

# Save label encoder for later use
with open('disease_label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Save the list of symptoms for the UI
with open('symptom_list.pkl', 'wb') as f:
    pickle.dump(list(column_names), f)

# Create a reverse mapping of disease codes to names
disease_mapping = dict(zip(le.transform(le.classes_), le.classes_))
with open('disease_mapping.pkl', 'wb') as f:
    pickle.dump(disease_mapping, f)

# Create a more accurate prediction function
def predict_disease(symptoms_list):
    """
    Predict disease based on a list of symptoms, using severity weights.
    
    Args:
        symptoms_list: List of symptom strings
    
    Returns:
        dict: Predicted disease and confidence score
    """
    # Create input vector with severity weights
    input_vector = np.zeros(len(column_names))
    for symptom in symptoms_list:
        if symptom in column_names:
            # Use severity if available, otherwise use default
            severity = severity_dict.get(symptom, default_severity)
            idx = column_names.get_loc(symptom)
            input_vector[idx] = severity
    
    # Get prediction probabilities
    probabilities = ensemble_model.predict_proba([input_vector])[0]
    max_prob_idx = np.argmax(probabilities)
    confidence = probabilities[max_prob_idx]
    disease_code = max_prob_idx
    disease_name = disease_mapping[disease_code]
    
    # Return top 3 predictions with confidence scores
    top_indices = probabilities.argsort()[-3:][::-1]
    top_predictions = [
        {"disease": disease_mapping[idx], "confidence": round(probabilities[idx] * 100, 2)}
        for idx in top_indices
    ]
    
    return {
        "predicted_disease": disease_name,
        "confidence": round(confidence * 100, 2),
        "top_predictions": top_predictions
    }

# Save the prediction function
with open('predict_disease.py', 'w') as f:
    f.write("""
import numpy as np
import pickle

# Load the models and data
ensemble_model = pickle.load(open('improved_ensemble_model.pkl', 'rb'))
disease_mapping = pickle.load(open('disease_mapping.pkl', 'rb')) 
column_names = pickle.load(open('symptom_list.pkl', 'rb'))

# Load severity dictionary
try:
    symptom_severity = pd.read_csv('training_data/Symptom-severity.csv')
    severity_dict = {}
    for index, row in symptom_severity.iterrows():
        symptom = row['Symptom'].strip().lower().replace(' ', '_')
        severity = row['weight']
        severity_dict[symptom] = severity
except:
    # Fallback severity dictionary if file isn't available
    severity_dict = {}
    default_severity = 5

def predict_disease(symptoms_list):
    # Create input vector with severity weights
    input_vector = np.zeros(len(column_names))
    for symptom in symptoms_list:
        if symptom in column_names:
            # Use severity if available, otherwise use default
            severity = severity_dict.get(symptom, 5)
            idx = column_names.index(symptom)
            input_vector[idx] = severity
    
    # Get prediction probabilities
    probabilities = ensemble_model.predict_proba([input_vector])[0]
    max_prob_idx = np.argmax(probabilities)
    confidence = probabilities[max_prob_idx]
    disease_code = max_prob_idx
    disease_name = disease_mapping[disease_code]
    
    # Return top 3 predictions with confidence scores
    top_indices = probabilities.argsort()[-3:][::-1]
    top_predictions = [
        {"disease": disease_mapping[idx], "confidence": round(probabilities[idx] * 100, 2)}
        for idx in top_indices
    ]
    
    return {
        "predicted_disease": disease_name,
        "confidence": round(confidence * 100, 2),
        "top_predictions": top_predictions
    }
""")

print("\nModel improvements complete. Run this script to generate the improved model.")
print("Use predict_disease() function to get predictions with confidence scores.") 