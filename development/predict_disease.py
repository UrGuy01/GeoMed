
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
