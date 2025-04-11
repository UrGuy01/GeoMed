import pickle
import numpy as np
import pandas as pd
import warnings

# Disable warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the models and data
print("Loading the improved ensemble model...")
try:
    ensemble_model = pickle.load(open('improved_ensemble_model.pkl', 'rb'))
    disease_mapping = pickle.load(open('disease_mapping.pkl', 'rb'))
    column_names = pickle.load(open('symptom_list.pkl', 'rb'))
    
    # Load severity dictionary
    symptom_severity = pd.read_csv('training_data/Symptom-severity.csv')
    severity_dict = {}
    for index, row in symptom_severity.iterrows():
        symptom = row['Symptom'].strip().lower().replace(' ', '_')
        severity = row['weight']
        severity_dict[symptom] = severity
    
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Define common symptom-disease associations to help with ambiguous cases
COMMON_DISEASE_SYMPTOMS = {
    "Common Cold": ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever", "headache", "runny_nose", "sinus_pressure"],
    "Influenza": ["high_fever", "headache", "chills", "fatigue", "joint_pain", "muscle_pain", "vomiting", "cough"],
    "Dengue": ["high_fever", "joint_pain", "muscle_pain", "fatigue", "skin_rash", "headache", "nausea", "loss_of_appetite"],
    "Malaria": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain", "fatigue"],
    "COVID-19": ["fever", "cough", "fatigue", "loss_of_smell", "loss_of_taste", "headache", "breathing_problems"],
    "Gastroenteritis": ["vomiting", "stomach_pain", "diarrhoea", "dehydration", "headache", "nausea"],
    "Migraine": ["headache", "nausea", "vomiting", "visual_disturbances", "pain_behind_the_eyes"],
    "Urinary Tract Infection": ["burning_micturition", "bladder_discomfort", "foul_smell_of urine", "continuous_feel_of_urine"]
}

# Common conditions prevalence weight (to prioritize common conditions over rare ones)
PREVALENCE_WEIGHTS = {
    "Common Cold": 1.5,
    "Influenza": 1.3,
    "Gastroenteritis": 1.2,
    "Urinary tract infection": 1.2,
    "Dengue": 1.1,
    "Malaria": 1.1,
    "Typhoid": 1.0,
    "Migraine": 1.1,
    "Acne": 1.1
}

def predict_disease(symptoms_list, min_confidence=15.0):
    """
    Predict disease based on a list of symptoms, using severity weights
    and common sense rules for ambiguous cases.
    
    Args:
        symptoms_list: List of symptom strings
        min_confidence: Minimum confidence threshold to consider a prediction reliable
    
    Returns:
        dict: Predicted disease and confidence score
    """
    # Create input vector with severity weights
    input_vector = np.zeros(len(column_names))
    for symptom in symptoms_list:
        if symptom in column_names:
            # Use severity if available, otherwise use default
            severity = severity_dict.get(symptom, 5)
            idx = column_names.index(symptom)
            input_vector[idx] = severity
        else:
            print(f"Warning: Symptom '{symptom}' not found in our database")
    
    # Get prediction probabilities from the model
    probabilities = ensemble_model.predict_proba([input_vector])[0]
    
    # Apply common sense rules for ambiguous cases
    
    # 1. Check if the symptoms closely match any known disease patterns
    common_disease_scores = {}
    for disease, disease_symptoms in COMMON_DISEASE_SYMPTOMS.items():
        # Calculate how many of the input symptoms match this disease's typical symptoms
        matching_symptoms = set(symptoms_list).intersection(set(disease_symptoms))
        match_score = len(matching_symptoms) / len(symptoms_list) if symptoms_list else 0
        
        # Calculate how comprehensive the match is (what % of typical symptoms are present)
        coverage_score = len(matching_symptoms) / len(disease_symptoms) if disease_symptoms else 0
        
        # Combined score (higher if more symptoms match and coverage is good)
        common_disease_scores[disease] = (match_score * 0.7 + coverage_score * 0.3) * 100
    
    # 2. Apply prevalence weights to model probabilities
    weighted_probabilities = probabilities.copy()
    for idx, disease_code in enumerate(range(len(weighted_probabilities))):
        disease_name = disease_mapping[disease_code]
        if disease_name in PREVALENCE_WEIGHTS:
            weighted_probabilities[idx] *= PREVALENCE_WEIGHTS[disease_name]
    
    # Get the model's top prediction
    max_prob_idx = np.argmax(weighted_probabilities)
    model_confidence = weighted_probabilities[max_prob_idx]
    model_disease_code = max_prob_idx
    model_disease = disease_mapping[model_disease_code]
    
    # 3. Determine the final prediction
    # If model confidence is very low and we have a good common disease match, use that instead
    best_common_disease = max(common_disease_scores.items(), key=lambda x: x[1]) if common_disease_scores else (None, 0)
    
    # Make the final prediction
    if model_confidence * 100 < min_confidence and best_common_disease[1] > 50:
        # Use the common disease pattern match as it's more reliable in this case
        final_disease = best_common_disease[0]
        final_confidence = best_common_disease[1]
        is_reliable = True
        confidence_source = "symptom pattern matching"
    else:
        # Use the model's prediction
        final_disease = model_disease
        final_confidence = model_confidence * 100
        is_reliable = final_confidence >= min_confidence
        confidence_source = "machine learning model"
    
    # Return top 3 predictions with confidence scores
    # For model predictions:
    top_indices = weighted_probabilities.argsort()[-3:][::-1]
    model_top_predictions = [
        {"disease": disease_mapping[idx], "confidence": round(weighted_probabilities[idx] * 100, 2)}
        for idx in top_indices
    ]
    
    # For common diseases:
    common_top_predictions = [
        {"disease": disease, "confidence": round(score, 2)}
        for disease, score in sorted(common_disease_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        if score > 20  # Only include if reasonable confidence
    ]
    
    # Return both sources of predictions
    return {
        "predicted_disease": final_disease,
        "confidence": round(final_confidence, 2),
        "reliability": "reliable" if is_reliable else "uncertain",
        "confidence_source": confidence_source,
        "model_predictions": model_top_predictions,
        "pattern_predictions": common_top_predictions
    }

def get_recommendations(disease):
    """Get medication recommendations for a disease"""
    try:
        medications_df = pd.read_csv('training_data/medications.csv')
        precautions_df = pd.read_csv('training_data/precautions_df.csv')
        descriptions_df = pd.read_csv('training_data/description.csv')
        diets_df = pd.read_csv('training_data/diets.csv')
        
        recommendations = {}
        
        # Get disease description
        desc = descriptions_df[descriptions_df['Disease'] == disease]['Description']
        recommendations['description'] = desc.iloc[0] if not desc.empty else "No description available"
        
        # Get precautions
        precautions = precautions_df[precautions_df['Disease'] == disease]
        precaution_list = []
        if not precautions.empty:
            for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                if col in precautions.columns and pd.notna(precautions[col].iloc[0]):
                    precaution_list.append(precautions[col].iloc[0])
        recommendations['precautions'] = precaution_list
        
        # Get medications
        meds = medications_df[medications_df['Disease'] == disease]['Medication']
        recommendations['medications'] = meds.iloc[0] if not meds.empty else "No specific medication data available"
        
        # Get diet recommendations
        diets = diets_df[diets_df['Disease'] == disease]['Diet']
        recommendations['diet'] = diets.iloc[0] if not diets.empty else "No specific diet data available"
        
        # If disease not found in our database, provide general advice
        if (desc.empty and not precaution_list and meds.empty and diets.empty):
            recommendations["description"] = "Detailed information not available for this condition. Please consult a healthcare provider."
            recommendations["precautions"] = ["Consult a healthcare provider", "Rest and stay hydrated", "Monitor your symptoms"]
            recommendations["medications"] = "Please consult a healthcare provider for appropriate medications."
            recommendations["diet"] = "A balanced diet and proper hydration are recommended."
        
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return {
            "description": "Error retrieving data",
            "precautions": [],
            "medications": "Error retrieving data",
            "diet": "Error retrieving data"
        }

def print_available_symptoms():
    """Print all available symptoms in an organized format"""
    print("\nAvailable symptoms (total: {}):\n".format(len(column_names)))
    
    # Print in columns for better readability
    symptoms_per_column = 3
    for i in range(0, len(column_names), symptoms_per_column):
        row = column_names[i:i+symptoms_per_column]
        formatted_row = "   ".join([f"{s.replace('_', ' '):<25}" for s in row])
        print(formatted_row)

def interactive_test():
    """Interactive mode to test the model with user input"""
    while True:
        print("\n" + "="*70)
        print(" GEOMED IMPROVED SYMPTOM CHECKER ".center(70, "="))
        print("="*70)
        
        print("\nEnter 'list' to see available symptoms")
        print("Enter 'quit' or 'exit' to end the program")
        
        # Get user input
        user_input = input("\nEnter your symptoms (comma-separated): ").strip().lower()
        
        if user_input in ['quit', 'exit']:
            break
        
        if user_input == 'list':
            print_available_symptoms()
            continue
        
        # Process symptoms
        symptoms = [s.strip().replace(' ', '_') for s in user_input.split(',')]
        
        if not symptoms or symptoms[0] == '':
            print("Please enter at least one symptom")
            continue
        
        print(f"\nAnalyzing symptoms: {', '.join([s.replace('_', ' ') for s in symptoms])}")
        
        try:
            # Get disease prediction
            result = predict_disease(symptoms)
            
            print("\n" + "-"*70)
            print(" PREDICTION RESULTS ".center(70, "-"))
            print("-"*70)
            
            print(f"\nPredicted Disease: {result['predicted_disease']}")
            print(f"Confidence: {result['confidence']}% ({result['reliability']})")
            print(f"Prediction method: {result['confidence_source']}")
            
            print("\nModel Top Predictions:")
            for idx, pred in enumerate(result['model_predictions'], 1):
                confidence_color = "\033[92m" if pred['confidence'] > 50 else "\033[93m" if pred['confidence'] > 20 else "\033[91m"
                print(f"  {idx}. {pred['disease']} ({confidence_color}{pred['confidence']}%\033[0m)")
            
            if result['pattern_predictions']:
                print("\nSymptom Pattern Matches:")
                for idx, pred in enumerate(result['pattern_predictions'], 1):
                    confidence_color = "\033[92m" if pred['confidence'] > 50 else "\033[93m" if pred['confidence'] > 20 else "\033[91m"
                    print(f"  {idx}. {pred['disease']} ({confidence_color}{pred['confidence']}%\033[0m)")
            
            # Get recommendations for the top prediction
            recommendations = get_recommendations(result['predicted_disease'])
            
            print("\n" + "-"*70)
            print(" MEDICAL RECOMMENDATIONS ".center(70, "-"))
            print("-"*70)
            
            print(f"\nDescription: {recommendations['description']}")
            
            print("\nPrecautions:")
            if recommendations['precautions']:
                for i, precaution in enumerate(recommendations['precautions'], 1):
                    print(f"  {i}. {precaution}")
            else:
                print("  No specific precautions found")
            
            print(f"\nRecommended Medications: {recommendations['medications']}")
            print(f"\nDiet Recommendations: {recommendations['diet']}")
            
            # Add disclaimer
            print("\n" + "-"*70)
            print("DISCLAIMER: This is not a substitute for professional medical advice.".center(70))
            print("If symptoms are severe or persistent, please consult a healthcare provider.".center(70))
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using the GeoMed Symptom Checker!")

if __name__ == "__main__":
    interactive_test() 