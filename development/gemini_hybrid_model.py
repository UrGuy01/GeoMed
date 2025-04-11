import pickle
import os
import sys
import numpy as np
import pandas as pd
import warnings
import json
import time
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path so we can import from models directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Disable warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables")

# Function to load model files
def load_models():
    try:
        print("Loading the improved ensemble model...")
        model_path = os.path.join('..', 'models', 'improved_ensemble_model.pkl')
        mapping_path = os.path.join('..', 'models', 'disease_mapping.pkl')
        symptoms_path = os.path.join('..', 'models', 'symptom_list.pkl')
        
        ensemble_model = pickle.load(open(model_path, 'rb'))
        disease_mapping = pickle.load(open(mapping_path, 'rb'))
        column_names = pickle.load(open(symptoms_path, 'rb'))
        
        # Load severity dictionary
        try:
            symptom_severity = pd.read_csv('../training_data/Symptom-severity.csv')
            severity_dict = {}
            for index, row in symptom_severity.iterrows():
                symptom = row['Symptom'].strip().lower().replace(' ', '_')
                severity = row['weight']
                severity_dict[symptom] = severity
        except Exception as e:
            print(f"Warning: Could not load severity data: {e}")
            severity_dict = {}
        
        # Load disease data for LLM context
        try:
            medications_df = pd.read_csv('../training_data/medications.csv')
            descriptions_df = pd.read_csv('../training_data/description.csv')
            diseases_list = list(descriptions_df['Disease'].unique())
        except Exception as e:
            print(f"Warning: Could not load disease data: {e}")
            medications_df = pd.DataFrame(columns=['Disease', 'Medication'])
            descriptions_df = pd.DataFrame(columns=['Disease', 'Description'])
            diseases_list = []
        
        print("Model loaded successfully!")
        return ensemble_model, disease_mapping, column_names, severity_dict, medications_df, descriptions_df, diseases_list
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None, None, None, None

# Load the models globally
try:
    ensemble_model, disease_mapping, column_names, severity_dict, medications_df, descriptions_df, diseases_list = load_models()
    
    # If models couldn't be loaded, create fallback data for demo purposes
    if ensemble_model is None:
        print("Using fallback symptom and disease data for demo")
        column_names = ["itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", 
                        "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue",
                        "muscle_wasting", "vomiting", "burning_micturition", "spotting_urination", "fatigue",
                        "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss"]
        
        disease_mapping = {0: "Fungal infection", 1: "Allergy", 2: "GERD", 3: "Chronic cholestasis", 
                          4: "Drug Reaction", 5: "Peptic ulcer disease", 6: "AIDS", 7: "Diabetes",
                          8: "Bronchial Asthma", 9: "Hypertension", 10: "Migraine", 11: "Cervical spondylosis"}
        
        severity_dict = {symptom: 5 for symptom in column_names}
        medications_df = pd.DataFrame(columns=['Disease', 'Medication'])
        descriptions_df = pd.DataFrame(columns=['Disease', 'Description'])
        diseases_list = list(disease_mapping.values())
except Exception as e:
    print(f"Error initializing model variables: {e}")
    sys.exit(1)

def gemini_diagnosis(symptoms, possible_diseases, model="gemini-1.5-pro"):
    """
    Get a diagnosis from Gemini based on symptoms.
    
    Args:
        symptoms: List of symptoms the patient is experiencing
        possible_diseases: List of potential diseases from the ML model
        model: Gemini model to use
    
    Returns:
        dict: LLM's diagnosis and confidence
    """
    if api_key:
        try:
            # Format symptoms and disease possibilities for the prompt
            symptom_text = ", ".join([s.replace('_', ' ') for s in symptoms])
            diseases_text = ", ".join([f"{d['disease']} ({d['confidence']}%)" for d in possible_diseases])
            
            # Create a detailed prompt for Gemini
            prompt = f"""As a medical AI assistant, analyze these symptoms: {symptom_text}

Based solely on these symptoms, what is the most likely diagnosis? Consider only these possible conditions: {", ".join([d["disease"] for d in possible_diseases])}

My machine learning model suggests these possibilities (with confidence scores):
{diseases_text}

Please provide:
1. The most likely diagnosis from the list
2. Your confidence level (0-100%)
3. Brief explanation for your diagnosis
4. Whether you agree with the ML model's top prediction

Format your response as JSON:
{{
  "diagnosis": "disease name",
  "confidence": confidence_percentage,
  "explanation": "your reasoning",
  "agrees_with_ml": true/false
}}
"""
            
            # Configure Gemini model
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 0,
                "max_output_tokens": 1024,
            }
            
            # Initialize the model
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config=generation_config
            )
            
            # Get the response
            response = model.generate_content(prompt)
            
            # Parse the response to extract JSON
            try:
                # Try to find JSON in the response
                content = response.text
                # Look for JSON content within {}
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    llm_result = json.loads(json_content)
                    
                    # Validate the required fields
                    required_fields = ["diagnosis", "confidence", "explanation", "agrees_with_ml"]
                    if all(field in llm_result for field in required_fields):
                        return llm_result
            except:
                pass
            
            # If JSON parsing failed or validation failed, use pattern matching to extract information
            content = response.text.lower()
            
            # Try to extract the diagnosis
            diagnosis = possible_diseases[0]["disease"]  # Default to ML's top prediction
            for d in possible_diseases:
                disease_name = d["disease"].lower()
                if f"diagnosis: {disease_name}" in content or f"diagnosis is {disease_name}" in content:
                    diagnosis = d["disease"]
                    break
            
            # Try to extract confidence
            confidence = 60  # Default confidence
            import re
            confidence_match = re.search(r'confidence:?\s*(\d+)', content)
            if confidence_match:
                confidence = min(100, max(0, int(confidence_match.group(1))))
            
            # Determine agreement with ML
            agrees_with_ml = diagnosis.lower() == possible_diseases[0]["disease"].lower()
            
            # Extract explanation or create one
            explanation = "Based on the symptoms provided, this appears to be the most likely diagnosis."
            explanation_match = re.search(r'explanation:?\s*([^\n]+)', content)
            if explanation_match:
                explanation = explanation_match.group(1)
            
            return {
                "diagnosis": diagnosis,
                "confidence": confidence,
                "explanation": explanation,
                "agrees_with_ml": agrees_with_ml
            }
                
        except Exception as e:
            print(f"Error with Gemini API call: {e}")
            return simulate_llm_diagnosis(symptoms, possible_diseases)
    else:
        # Fallback to simulation if no project ID
        print("No Google API Key found. Using simulated diagnosis.")
        return simulate_llm_diagnosis(symptoms, possible_diseases)

def simulate_llm_diagnosis(symptoms, possible_diseases):
    """Simulate LLM diagnosis when API is not available"""
    # This is a simplified simulation of what an LLM might return
    # In real use, replace this with actual API calls
    
    # Define some basic symptom-disease associations for simulation
    symptom_disease_map = {
        "high_fever": ["Influenza", "Malaria", "Typhoid", "Dengue"],
        "joint_pain": ["Arthritis", "Dengue", "Chikungunya", "Typhoid"],
        "headache": ["Migraine", "Common Cold", "Influenza", "Malaria"],
        "cough": ["Common Cold", "Bronchitis", "Pneumonia", "Tuberculosis"],
        "skin_rash": ["Fungal infection", "Chickenpox", "Measles", "Psoriasis"],
        "fatigue": ["Anemia", "Influenza", "Depression", "Hypothyroidism"],
        "chest_pain": ["Heart attack", "GERD", "Bronchial Asthma"],
        "stomach_pain": ["Gastroenteritis", "GERD", "Peptic ulcer diseae"],
        "vomiting": ["Gastroenteritis", "Food Poisoning", "Hepatitis"],
        "shivering": ["Malaria", "Influenza", "Common Cold"],
        "diarrhoea": ["Gastroenteritis", "Food Poisoning", "Typhoid"],
        "continuous_sneezing": ["Common Cold", "Allergic rhinitis", "Sinusitis"]
    }
    
    # Count matches for each disease in the possible diseases list
    disease_scores = {d["disease"]: 0 for d in possible_diseases}
    
    for symptom in symptoms:
        related_diseases = symptom_disease_map.get(symptom, [])
        for disease in related_diseases:
            if disease in disease_scores:
                disease_scores[disease] += 1
    
    # Look at ML confidence scores as well
    for d in possible_diseases:
        disease_scores[d["disease"]] += d["confidence"] / 20  # Convert confidence to points
    
    # Find the best match
    if disease_scores:
        best_disease = max(disease_scores.items(), key=lambda x: x[1])
        ml_top_disease = possible_diseases[0]["disease"]
        
        confidence = min(90, best_disease[1] * 10 + 50)  # Scale to reasonable confidence
        
        return {
            "diagnosis": best_disease[0],
            "confidence": confidence,
            "explanation": f"Based on the symptoms {', '.join(symptoms)}, {best_disease[0]} is the most likely diagnosis.",
            "agrees_with_ml": best_disease[0] == ml_top_disease
        }
    else:
        # Fallback to ML model's top prediction
        return {
            "diagnosis": possible_diseases[0]["disease"],
            "confidence": possible_diseases[0]["confidence"],
            "explanation": "Insufficient symptom information to make a confident diagnosis.",
            "agrees_with_ml": True
        }

def predict_disease_hybrid(symptoms_list, min_confidence=15.0):
    """
    Hybrid prediction that combines ML model with PaLM 2 analysis.
    
    Args:
        symptoms_list: List of symptom strings
        min_confidence: Minimum confidence threshold
    
    Returns:
        dict: Combined prediction results
    """
    # Step 1: Get ML model prediction
    ml_prediction = predict_disease_ml(symptoms_list, min_confidence)
    
    # Step 2: Get PaLM 2 prediction
    llm_result = gemini_diagnosis(
        symptoms_list, 
        ml_prediction["model_predictions"]
    )
    
    # Step 3: Determine final prediction
    # If ML confidence is high (>70%) and LLM agrees, use ML prediction
    if ml_prediction["confidence"] > 70 and llm_result["agrees_with_ml"]:
        final_prediction = ml_prediction["predicted_disease"]
        final_confidence = ml_prediction["confidence"]
        prediction_source = "ML model (high confidence, Gemini agrees)"
    
    # If ML confidence is medium (>40%) and LLM agrees, use ML prediction
    elif ml_prediction["confidence"] > 40 and llm_result["agrees_with_ml"]:
        final_prediction = ml_prediction["predicted_disease"]
        final_confidence = (ml_prediction["confidence"] + llm_result["confidence"]) / 2
        prediction_source = "ML model (medium confidence, Gemini agrees)"
    
    # If LLM confidence is high (>70%), prefer LLM prediction
    elif llm_result["confidence"] > 70:
        final_prediction = llm_result["diagnosis"]
        final_confidence = llm_result["confidence"]
        prediction_source = "MedLM (high confidence)"
    
    # In case of disagreement and similar confidence, average the results
    elif abs(ml_prediction["confidence"] - llm_result["confidence"]) < 15:
        # Use the one with slightly higher confidence
        if ml_prediction["confidence"] >= llm_result["confidence"]:
            final_prediction = ml_prediction["predicted_disease"]
            final_confidence = ml_prediction["confidence"]
            prediction_source = "ML model (slight advantage in confidence)"
        else:
            final_prediction = llm_result["diagnosis"]
            final_confidence = llm_result["confidence"]
            prediction_source = "MedLM (slight advantage in confidence)"
    
    # In case of low confidence all around, use pattern matching
    elif ml_prediction["pattern_predictions"] and ml_prediction["pattern_predictions"][0]["confidence"] > 40:
        final_prediction = ml_prediction["pattern_predictions"][0]["disease"]
        final_confidence = ml_prediction["pattern_predictions"][0]["confidence"]
        prediction_source = "Pattern matching (low ML and MedLM confidence)"
    
    # Default to LLM for ambiguous cases
    else:
        final_prediction = llm_result["diagnosis"]
        final_confidence = llm_result["confidence"]
        prediction_source = "MedLM (default for ambiguous case)"
    
    return {
        "predicted_disease": final_prediction,
        "confidence": final_confidence,
        "reliability": "reliable" if final_confidence >= min_confidence else "uncertain",
        "prediction_source": prediction_source,
        "ml_prediction": ml_prediction,
        "llm_prediction": llm_result
    }

def predict_disease_ml(symptoms_list, min_confidence=15.0):
    """
    Predict disease based on ML model with severity weights and pattern matching.
    This is the original ML-only prediction function.
    
    Args:
        symptoms_list: List of symptom strings
        min_confidence: Minimum confidence threshold
    
    Returns:
        dict: Prediction results
    """
    # Common disease patterns for pattern matching
    COMMON_DISEASE_SYMPTOMS = {
        "Common Cold": ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever", "headache", "runny_nose", "sinus_pressure"],
        "Influenza": ["high_fever", "headache", "chills", "fatigue", "joint_pain", "muscle_pain", "vomiting", "cough"],
        "Dengue": ["high_fever", "joint_pain", "muscle_pain", "fatigue", "skin_rash", "headache", "nausea", "loss_of_appetite"],
        "Malaria": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain", "fatigue"],
        "COVID-19": ["high_fever", "cough", "fatigue", "loss_of_smell", "loss_of_taste", "headache", "breathing_problems"],
        "Gastroenteritis": ["vomiting", "stomach_pain", "diarrhoea", "dehydration", "headache", "nausea"],
        "Migraine": ["headache", "nausea", "vomiting", "visual_disturbances", "pain_behind_the_eyes"],
        "Urinary Tract Infection": ["burning_micturition", "bladder_discomfort", "foul_smell_of urine", "continuous_feel_of_urine"]
    }
    
    # Prevalence weights for common conditions
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
    
    # Get model prediction probabilities
    probabilities = ensemble_model.predict_proba([input_vector])[0]
    
    # Check for pattern matches
    common_disease_scores = {}
    for disease, disease_symptoms in COMMON_DISEASE_SYMPTOMS.items():
        # Calculate how many of the input symptoms match this disease's typical symptoms
        matching_symptoms = set(symptoms_list).intersection(set(disease_symptoms))
        match_score = len(matching_symptoms) / len(symptoms_list) if symptoms_list else 0
        
        # Calculate how comprehensive the match is
        coverage_score = len(matching_symptoms) / len(disease_symptoms) if disease_symptoms else 0
        
        # Combined score
        common_disease_scores[disease] = (match_score * 0.7 + coverage_score * 0.3) * 100
    
    # Apply prevalence weights to model probabilities
    weighted_probabilities = probabilities.copy()
    for idx, disease_code in enumerate(range(len(weighted_probabilities))):
        disease_name = disease_mapping[disease_code]
        if disease_name in PREVALENCE_WEIGHTS:
            weighted_probabilities[idx] *= PREVALENCE_WEIGHTS[disease_name]
    
    # Get the model's top prediction
    max_prob_idx = np.argmax(weighted_probabilities)
    model_confidence = weighted_probabilities[max_prob_idx]
    model_disease = disease_mapping[max_prob_idx]
    
    # Determine the final prediction
    best_common_disease = max(common_disease_scores.items(), key=lambda x: x[1]) if common_disease_scores else (None, 0)
    
    # If model confidence is low and pattern match is good, use pattern match
    if model_confidence * 100 < min_confidence and best_common_disease[1] > 50:
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
    
    # Return top predictions
    top_indices = weighted_probabilities.argsort()[-3:][::-1]
    model_top_predictions = [
        {"disease": disease_mapping[idx], "confidence": round(weighted_probabilities[idx] * 100, 2)}
        for idx in top_indices
    ]
    
    common_top_predictions = [
        {"disease": disease, "confidence": round(score, 2)}
        for disease, score in sorted(common_disease_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        if score > 20
    ]
    
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
        # Try to use the global variables first if they were loaded successfully
        global medications_df, descriptions_df
        
        recommendations = {}
        
        # Get disease description
        if 'descriptions_df' in globals() and not descriptions_df.empty:
            desc = descriptions_df[descriptions_df['Disease'] == disease]['Description']
            recommendations['description'] = desc.iloc[0] if not desc.empty else "No description available"
        else:
            recommendations['description'] = "No description available for this condition"
        
        # Get precautions
        try:
            precautions_df = pd.read_csv('../training_data/precautions_df.csv')
            precautions = precautions_df[precautions_df['Disease'] == disease]
            precaution_list = []
            if not precautions.empty:
                for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                    if col in precautions.columns and pd.notna(precautions[col].iloc[0]):
                        precaution_list.append(precautions[col].iloc[0])
            recommendations['precautions'] = precaution_list
        except Exception as e:
            print(f"Warning: Could not load precautions data: {e}")
            recommendations['precautions'] = ["Rest and stay hydrated", "Monitor symptoms", "Consult a healthcare provider"]
        
        # Get medications
        if 'medications_df' in globals() and not medications_df.empty:
            meds = medications_df[medications_df['Disease'] == disease]['Medication']
            recommendations['medications'] = meds.iloc[0] if not meds.empty else "No specific medication data available"
        else:
            recommendations['medications'] = "No specific medication data available"
        
        # Get diet recommendations
        try:
            diets_df = pd.read_csv('../training_data/diets.csv')
            diets = diets_df[diets_df['Disease'] == disease]['Diet']
            recommendations['diet'] = diets.iloc[0] if not diets.empty else "No specific diet data available"
        except Exception as e:
            print(f"Warning: Could not load diet data: {e}")
            recommendations['diet'] = "A balanced diet and proper hydration are recommended"
        
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return {
            "description": "Information not available for this condition. Please consult a healthcare provider.",
            "precautions": ["Consult a healthcare provider", "Rest and stay hydrated", "Monitor your symptoms"],
            "medications": "Please consult a healthcare provider for appropriate medications.",
            "diet": "A balanced diet and proper hydration are recommended."
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
    """Interactive mode to test the hybrid model with user input"""
    print("\nWelcome to the GeoMed Hybrid Diagnosis System")
    print("This system combines ML predictions with Google Gemini for better accuracy")
    
    if not api_key:
        print("\nNote: Google API Key not found. Using simulated Gemini responses.")
        print("To use real Gemini API, set your GOOGLE_API_KEY environment variable.")
    
    while True:
        print("\n" + "="*70)
        print(" GEOMED HYBRID ML-MEDLM SYMPTOM CHECKER ".center(70, "="))
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
        print("Running ML model and consulting MedLM...")
        
        start_time = time.time()
        
        try:
            # Get hybrid prediction
            result = predict_disease_hybrid(symptoms)
            
            elapsed_time = time.time() - start_time
            
            print("\n" + "-"*70)
            print(" HYBRID PREDICTION RESULTS ".center(70, "-"))
            print("-"*70)
            
            print(f"\nFinal Diagnosis: {result['predicted_disease']}")
            print(f"Confidence: {result['confidence']}% ({result['reliability']})")
            print(f"Source: {result['prediction_source']}")
            print(f"Analysis completed in {elapsed_time:.2f} seconds")
            
            print("\nML Model Predictions:")
            for idx, pred in enumerate(result['ml_prediction']['model_predictions'], 1):
                confidence_color = "\033[92m" if pred['confidence'] > 50 else "\033[93m" if pred['confidence'] > 20 else "\033[91m"
                print(f"  {idx}. {pred['disease']} ({confidence_color}{pred['confidence']}%\033[0m)")
            
            print("\nMedLM Analysis:")
            print(f"  Diagnosis: {result['llm_prediction']['diagnosis']}")
            print(f"  Confidence: {result['llm_prediction']['confidence']}%")
            print(f"  Explanation: {result['llm_prediction']['explanation']}")
            print(f"  Agrees with ML: {'Yes' if result['llm_prediction']['agrees_with_ml'] else 'No'}")
            
            if result['ml_prediction']['pattern_predictions']:
                print("\nSymptom Pattern Matches:")
                for idx, pred in enumerate(result['ml_prediction']['pattern_predictions'], 1):
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
    
    print("\nThank you for using the GeoMed Hybrid Diagnosis System!")

if __name__ == "__main__":
    interactive_test() 