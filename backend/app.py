import pickle
import os
import sys
import numpy as np
import pandas as pd
import warnings
import json
import time
import google.generativeai as genai
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
import math
import random

# Add parent directory to path so we can import from models directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Disable warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables")

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase = None

if supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key)
        
        # Test the connection by making a simple query
        test_response = supabase.table('symptoms_data').select('count(*)', count='exact').limit(1).execute()
        
        if hasattr(test_response, 'error') and test_response.error:
            print(f"Error connecting to Supabase: {test_response.error}")
            print("Database will run in mock mode (no data persistence)")
            supabase = None
        else:
            count = test_response.count if hasattr(test_response, 'count') else 0
            print(f"Successfully connected to Supabase. Found {count} records in symptoms_data table.")
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        print("Database will run in mock mode (no data persistence)")
        import traceback
        traceback.print_exc()
        supabase = None
else:
    print("Warning: Supabase credentials not found in environment variables")
    print("Database will run in mock mode (no data persistence)")
    supabase = None

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
                        return llm_result["diagnosis"], llm_result["confidence"], llm_result["explanation"]
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
            
            # Extract explanation or create one
            explanation = "Based on the symptoms provided, this appears to be the most likely diagnosis."
            explanation_match = re.search(r'explanation:?\s*([^\n]+)', content)
            if explanation_match:
                explanation = explanation_match.group(1)
            
            return diagnosis, confidence, explanation
                
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
        
        confidence = min(90, best_disease[1] * 10 + 50)  # Scale to reasonable confidence
        explanation = f"Based on the symptoms {', '.join(symptoms)}, {best_disease[0]} is the most likely diagnosis."
        
        return best_disease[0], confidence, explanation
    else:
        # Fallback to ML model's top prediction
        return possible_diseases[0]["disease"], possible_diseases[0]["confidence"], "Insufficient symptom information to make a confident diagnosis."

def hybrid_diagnosis(symptoms):
    """
    Perform hybrid diagnosis combining ML and Gemini
    """
    print(f"Diagnosing symptoms: {', '.join(symptoms)}")
    
    # First get ML prediction
    ml_disease, ml_confidence, possible_diseases = predict_disease_ml(symptoms)
    print(f"ML diagnosis: {ml_disease} with {ml_confidence:.1f}% confidence")
    
    # Then get Gemini/LLM prediction
    llm_disease, llm_confidence, explanation = gemini_diagnosis(symptoms, possible_diseases)
    print(f"LLM diagnosis: {llm_disease} with {llm_confidence:.1f}% confidence")
    
    return ml_disease, ml_confidence, llm_disease, llm_confidence, explanation

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
        Tuple: (predicted_disease, confidence, possible_diseases)
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
    
    try:
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
        
        # Apply prevalence weights to model probabilities
        weighted_probabilities = probabilities.copy()
        for idx, disease_code in enumerate(range(len(weighted_probabilities))):
            disease_name = disease_mapping[disease_code]
            if disease_name in PREVALENCE_WEIGHTS:
                weighted_probabilities[idx] *= PREVALENCE_WEIGHTS[disease_name]
        
        # Get the model's top prediction
        max_prob_idx = np.argmax(weighted_probabilities)
        model_confidence = weighted_probabilities[max_prob_idx] * 100
        model_disease = disease_mapping[max_prob_idx]
        
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
        
        # If model confidence is low and pattern match is good, use pattern match
        best_common_disease = max(common_disease_scores.items(), key=lambda x: x[1]) if common_disease_scores else (None, 0)
        if model_confidence < min_confidence and best_common_disease[1] > 50:
            final_disease = best_common_disease[0]
            final_confidence = best_common_disease[1]
        else:
            # Use the model's prediction
            final_disease = model_disease
            final_confidence = model_confidence
        
        # Return top predictions for the LLM
        top_indices = weighted_probabilities.argsort()[-3:][::-1]
        model_top_predictions = [
            {"disease": disease_mapping[idx], "confidence": round(weighted_probabilities[idx] * 100, 2)}
            for idx in top_indices
        ]
        
        return final_disease, final_confidence, model_top_predictions
    
    except Exception as e:
        print(f"Error in predict_disease_ml: {e}")
        # Fallback to a simple prediction when the model fails
        fallback_disease = list(disease_mapping.values())[0]
        fallback_predictions = [
            {"disease": fallback_disease, "confidence": 40.0},
            {"disease": list(disease_mapping.values())[1], "confidence": 30.0},
            {"disease": list(disease_mapping.values())[2], "confidence": 20.0}
        ]
        return fallback_disease, 40.0, fallback_predictions

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
            # Get hybrid prediction using the hybrid_diagnosis function
            ml_disease, ml_confidence, llm_disease, llm_confidence, explanation = hybrid_diagnosis(symptoms)
            
            elapsed_time = time.time() - start_time
            
            print("\n" + "-"*70)
            print(" HYBRID PREDICTION RESULTS ".center(70, "-"))
            print("-"*70)
            
            # Determine final diagnosis based on confidence levels
            if ml_confidence > 70 and ml_disease == llm_disease:
                final_diagnosis = ml_disease
                final_confidence = ml_confidence
                source = "ML model with Gemini validation"
            elif llm_confidence > 70:
                final_diagnosis = llm_disease
                final_confidence = llm_confidence
                source = "Gemini with high confidence"
            elif ml_confidence > llm_confidence:
                final_diagnosis = ml_disease
                final_confidence = ml_confidence
                source = "ML model (higher confidence than Gemini)"
            else:
                final_diagnosis = llm_disease
                final_confidence = llm_confidence
                source = "Gemini (higher confidence than ML model)"
            
            print(f"\nFinal Diagnosis: {final_diagnosis}")
            print(f"Confidence: {final_confidence:.1f}% ({final_confidence >= 50 and 'reliable' or 'uncertain'})")
            print(f"Source: {source}")
            print(f"Analysis completed in {elapsed_time:.2f} seconds")
            
            print("\nML Model Prediction:")
            print(f"  Diagnosis: {ml_disease}")
            print(f"  Confidence: {ml_confidence:.1f}%")
            
            print("\nMedLM Analysis:")
            print(f"  Diagnosis: {llm_disease}")
            print(f"  Confidence: {llm_confidence:.1f}%")
            print(f"  Explanation: {explanation}")
            
            # Get recommendations for the final diagnosis
            recommendations = get_recommendations(final_diagnosis)
            
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

# API Routes
@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    try:
        print("=== Received diagnosis request ===")
        data = request.json
        symptoms = data.get('symptoms', [])
        location = data.get('location')
        
        print(f"Symptoms: {symptoms}")
        print(f"Location data: {location}")
        
        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400
        
        # Get current timestamp
        timestamp = datetime.utcnow().isoformat()
        
        # Get diagnosis using the hybrid model
        ml_disease, ml_confidence, llm_disease, llm_confidence, explanation = hybrid_diagnosis(symptoms)
        
        # Determine the final diagnosis to return
        # Use LLM diagnosis if it has higher confidence or if ML confidence is low
        if llm_confidence >= ml_confidence or ml_confidence < 50:
            final_diagnosis = llm_disease
            final_confidence = llm_confidence
            diagnosis_source = "AI Assistant"
        else:
            final_diagnosis = ml_disease
            final_confidence = ml_confidence
            diagnosis_source = "ML Model"
        
        # Get recommendations for the diagnosed disease
        recommendations = get_recommendations(final_diagnosis)
        
        # Store data in Supabase if available
        if supabase is not None:
            try:
                print("=== Attempting to store data in Supabase ===")
                # Prepare data for storage
                symptom_data = {
                    'symptoms': symptoms,
                    'diagnosis': final_diagnosis,
                    'confidence': float(final_confidence),
                    'timestamp': timestamp,
                    'source': diagnosis_source
                }
                
                # Ensure symptoms is properly formatted as a list for Postgres array
                if isinstance(symptoms, list):
                    print(f"Symptoms list format: {symptoms}")
                else:
                    # Convert to list if not already
                    symptoms_list = list(symptoms) if hasattr(symptoms, '__iter__') else [str(symptoms)]
                    symptom_data['symptoms'] = symptoms_list
                    print(f"Converted symptoms to list: {symptoms_list}")
                
                print(f"Preparing to save data to Supabase: {symptom_data}")
                
                # Add location data if available
                if location:
                    print(f"Adding location data: {location}")
                    symptom_data.update({
                        'latitude': location.get('latitude'),
                        'longitude': location.get('longitude'),
                        'accuracy': location.get('accuracy'),
                        'location_source': location.get('source'),
                    })
                    
                    # Add additional IP-based location details if available
                    if location.get('source') == 'ip':
                        symptom_data.update({
                            'ip_address': location.get('ip'),
                            'city': location.get('city'),
                            'region': location.get('region'),
                            'country': location.get('country')
                        })
                
                # Insert data into Supabase
                print("Sending data to Supabase...")
                print(f"Supabase instance: {type(supabase)}")
                
                # Sanitize and verify the data
                for key, value in symptom_data.items():
                    print(f"Key: {key}, Value type: {type(value)}, Value: {value}")
                
                try:
                    response = supabase.table('symptoms_data').insert(symptom_data).execute()
                    print(f"Raw Supabase response: {response}")
                    
                    if hasattr(response, 'error') and response.error:
                        print(f"Error storing data in Supabase: {response.error}")
                    else:
                        print(f"Successfully stored symptom data in Supabase")
                except Exception as insert_error:
                    print(f"Exception during Supabase insert: {insert_error}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as db_error:
                print(f"Error preparing data for Supabase: {db_error}")
                import traceback
                traceback.print_exc()
        else:
            print("Supabase connection not available - skipping data storage")
            
        # Prepare result
        result = {
            'diagnosis': final_diagnosis,
            'confidence': float(final_confidence),
            'explanation': explanation,
            'source': diagnosis_source,
            'recommendations': {
                'description': recommendations['description'],
                'precautions': recommendations['precautions'],
                'medications': recommendations['medications'],
                'diet': recommendations['diet']
            }
        }
        
        # If confidence is low, suggest more symptoms to check and alternative diagnoses
        if final_confidence < 40:
            suggested_info = get_suggested_info(symptoms, final_diagnosis)
            result['suggested_symptoms'] = suggested_info['suggested_symptoms']
            result['alternative_diagnoses'] = suggested_info['alternative_diagnoses']
            result['needs_more_info'] = True
        else:
            result['needs_more_info'] = False
        
        print(f"Final diagnosis: {final_diagnosis} ({diagnosis_source}) with {final_confidence:.1f}% confidence")
        return jsonify(result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error in diagnose endpoint: {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    try:
        # Return the list of symptoms from the model
        return jsonify({
            'symptoms': column_names
        })
    except Exception as e:
        error_msg = str(e)
        print(f"Error in symptoms endpoint: {error_msg}")
        return jsonify({'error': error_msg}), 500

def get_suggested_info(symptoms, current_diagnosis):
    """
    Get suggested symptoms and alternative diagnoses using Gemini API
    when confidence is low
    
    Args:
        symptoms: List of current symptoms
        current_diagnosis: Current low-confidence diagnosis
        
    Returns:
        dict: Containing suggested symptoms to check and alternative diagnoses
    """
    if not api_key:
        # Fallback suggestions if API key is not available
        return {
            'suggested_symptoms': ["fever", "cough", "headache", "fatigue", "nausea"],
            'alternative_diagnoses': ["Common Cold", "Influenza", "Allergic Reaction"]
        }
    
    try:
        # Format symptoms for the prompt
        symptom_text = ", ".join([s.replace('_', ' ') for s in symptoms])
        
        # Create a detailed prompt for Gemini
        prompt = f"""As a medical AI assistant, I need your help with a patient case.

The patient has the following symptoms: {symptom_text}

My initial diagnosis is {current_diagnosis}, but I have low confidence in this assessment.

Please provide:
1. Five additional symptoms that would be important to check for to refine this diagnosis
2. Three alternative potential diagnoses that could explain these symptoms
3. Format your response as JSON:

{{
  "suggested_symptoms": ["symptom1", "symptom2", "symptom3", "symptom4", "symptom5"],
  "alternative_diagnoses": [
    {{"disease": "disease1", "key_symptoms": "brief list of key symptoms"}},
    {{"disease": "disease2", "key_symptoms": "brief list of key symptoms"}},
    {{"disease": "disease3", "key_symptoms": "brief list of key symptoms"}}
  ]
}}
"""
        
        # Configure Gemini model
        generation_config = {
            "temperature": 0.2,  # Low temperature for more focused responses
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
        content = response.text
        
        # Parse the response to extract JSON
        try:
            # Try to find JSON in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                result = json.loads(json_content)
                
                # Extract and format the suggested symptoms
                suggested_symptoms = [s.strip().lower().replace(' ', '_') for s in result.get('suggested_symptoms', [])]
                
                # Extract and format the alternative diagnoses
                alternative_diagnoses = result.get('alternative_diagnoses', [])
                if isinstance(alternative_diagnoses, list):
                    formatted_alternatives = []
                    for alt in alternative_diagnoses:
                        if isinstance(alt, dict) and 'disease' in alt:
                            formatted_alternatives.append({
                                'disease': alt['disease'],
                                'key_symptoms': alt.get('key_symptoms', '')
                            })
                    
                    return {
                        'suggested_symptoms': suggested_symptoms[:5],  # Limit to 5
                        'alternative_diagnoses': formatted_alternatives[:3]  # Limit to 3
                    }
        except Exception as e:
            print(f"Error parsing Gemini suggestion response: {e}")
        
        # Fallback if JSON parsing failed
        return {
            'suggested_symptoms': ["fever", "headache", "fatigue", "cough", "sore_throat"],
            'alternative_diagnoses': [
                {"disease": "Common Cold", "key_symptoms": "runny nose, sore throat, cough"},
                {"disease": "Influenza", "key_symptoms": "fever, body aches, fatigue"},
                {"disease": "Allergic Reaction", "key_symptoms": "itching, rash, runny nose"}
            ]
        }
    except Exception as e:
        print(f"Error getting suggestions from Gemini: {e}")
        return {
            'suggested_symptoms': ["fever", "cough", "headache", "fatigue", "nausea"],
            'alternative_diagnoses': [
                {"disease": "Common Cold", "key_symptoms": "runny nose, sore throat, cough"},
                {"disease": "Influenza", "key_symptoms": "fever, body aches, fatigue"},
                {"disease": "Allergic Reaction", "key_symptoms": "itching, rash, runny nose"}
            ]
        }

@app.route('/api/geo-data', methods=['GET'])
def get_geo_data():
    """
    Get geographic distribution of symptoms for visualization
    Query params:
    - days: Number of days to look back (default: 30)
    - symptoms: Comma-separated list of symptoms to filter (optional)
    - diagnosis: Specific diagnosis to filter (optional)
    """
    try:
        # Get query parameters
        days = request.args.get('days', 30, type=int)
        symptoms_filter = request.args.get('symptoms', '')
        diagnosis_filter = request.args.get('diagnosis', '')
        
        # Calculate the start date
        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        if supabase is None:
            # If no Supabase connection, return mock data
            return jsonify({
                'data': generate_mock_geo_data(days, symptoms_filter, diagnosis_filter),
                'source': 'mock'
            })
        
        try:
            # Query Supabase for geo data
            query = supabase.table('symptoms_data').select('*').gte('timestamp', start_date)
            
            # Apply filters if provided
            if symptoms_filter:
                symptoms_list = [s.strip() for s in symptoms_filter.split(',')]
                # This is a simplified filter approach - actual implementation would depend on how symptoms are stored
                for symptom in symptoms_list:
                    query = query.contains('symptoms', [symptom])
                    
            if diagnosis_filter:
                query = query.eq('diagnosis', diagnosis_filter)
            
            # Execute the query
            response = query.execute()
            
            if hasattr(response, 'error') and response.error:
                print(f"Error querying data from Supabase: {response.error}")
                return jsonify({'error': 'Database query error'}), 500
            
            # Process the results
            geo_data = []
            if hasattr(response, 'data'):
                for item in response.data:
                    # Only include items with location data
                    if 'latitude' in item and 'longitude' in item:
                        geo_data.append({
                            'lat': item['latitude'],
                            'lng': item['longitude'],
                            'symptoms': item['symptoms'],
                            'diagnosis': item['diagnosis'],
                            'timestamp': item['timestamp'],
                            'weight': 1  # Can be modified based on criteria like confidence
                        })
            
            return jsonify({
                'data': geo_data,
                'source': 'supabase'
            })
            
        except Exception as db_error:
            print(f"Error retrieving geo data from Supabase: {db_error}")
            return jsonify({'error': str(db_error)}), 500
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error in geo-data endpoint: {error_msg}")
        return jsonify({'error': error_msg}), 500
        
def generate_mock_geo_data(days=30, symptoms_filter='', diagnosis_filter=''):
    """Generate mock geographic data for demonstration"""
    mock_data = []
    
    # Number of data points
    num_points = min(500, days * 5)  # Up to 5 points per day, max 500
    
    # Common disease clusters in different regions
    clusters = [
        # North America - Flu cluster
        {'center': [40.7128, -74.0060], 'radius': 5.0, 'symptoms': ['cough', 'fever', 'headache'],
         'diagnosis': 'Influenza', 'count': int(num_points * 0.3)},
        # Europe - Cold cluster
        {'center': [51.5074, -0.1278], 'radius': 4.0, 'symptoms': ['runny_nose', 'sore_throat', 'cough'],
         'diagnosis': 'Common Cold', 'count': int(num_points * 0.25)},
        # Asia - Dengue cluster
        {'center': [13.7563, 100.5018], 'radius': 6.0, 'symptoms': ['high_fever', 'headache', 'joint_pain'],
         'diagnosis': 'Dengue', 'count': int(num_points * 0.2)},
        # Random points globally
        {'center': [0, 0], 'radius': 180.0, 'symptoms': ['fatigue', 'headache', 'nausea'],
         'diagnosis': 'Various', 'count': int(num_points * 0.25)}
    ]
    
    # Generate points for each cluster
    for cluster in clusters:
        # Apply filters
        if symptoms_filter:
            symptoms_list = [s.strip() for s in symptoms_filter.split(',')]
            if not any(s in cluster['symptoms'] for s in symptoms_list):
                continue
                
        if diagnosis_filter and cluster['diagnosis'] != diagnosis_filter and cluster['diagnosis'] != 'Various':
            continue
            
        # Generate points
        for _ in range(cluster['count']):
            # Random point within the cluster radius
            angle = random.uniform(0, 2 * 3.14159)
            distance = random.uniform(0, cluster['radius'])
            
            # Calculate coordinates (simplified)
            lat = cluster['center'][0] + distance * math.cos(angle)
            lng = cluster['center'][1] + distance * math.sin(angle)
            
            # Limit to valid coordinates
            lat = max(-90, min(90, lat))
            lng = max(-180, min(180, lng))
            
            # Random timestamp within the specified days
            days_ago = random.uniform(0, days)
            timestamp = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
            
            # Choose a diagnosis
            if cluster['diagnosis'] == 'Various':
                diagnosis = random.choice(list(disease_mapping.values()))
            else:
                diagnosis = cluster['diagnosis']
                
            # Choose symptoms
            if random.random() < 0.8:
                # Use cluster symptoms
                symptoms = cluster['symptoms'].copy()
                # Add some random symptoms
                additional = random.sample(column_names, min(2, len(column_names)))
                symptoms.extend(additional)
            else:
                # Completely random symptoms
                symptoms = random.sample(column_names, min(5, len(column_names)))
                
            mock_data.append({
                'lat': lat,
                'lng': lng,
                'symptoms': symptoms,
                'diagnosis': diagnosis,
                'timestamp': timestamp,
                'weight': random.uniform(0.5, 1.5)
            })
    
    return mock_data

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    try:
        # Get date range from query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # If no dates provided, default to last 30 days
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
        if not end_date:
            end_date = datetime.utcnow().isoformat()
        
        if supabase is None:
            # Generate mock data for demonstration if no database connection
            diagnoses = generate_mock_diagnoses(start_date, end_date)
            data_source = 'mock'
        else:
            try:
                # Query Supabase for real data
                query = supabase.table('symptoms_data').select('*')
                
                if start_date:
                    query = query.gte('timestamp', start_date)
                if end_date:
                    query = query.lte('timestamp', end_date)
                
                result = query.execute()
                
                if hasattr(result, 'error') and result.error:
                    print(f"Error querying statistics from Supabase: {result.error}")
                    # Fall back to mock data
                    diagnoses = generate_mock_diagnoses(start_date, end_date)
                    data_source = 'mock (database error)'
                elif hasattr(result, 'data'):
                    # Transform the data into the expected format
                    diagnoses = []
                    for item in result.data:
                        diagnoses.append({
                            'symptoms': item.get('symptoms', []),
                            'ml_diagnosis': item.get('diagnosis', 'Unknown'),
                            'ml_confidence': item.get('confidence', 0),
                            'latitude': item.get('latitude'),
                            'longitude': item.get('longitude'),
                            'timestamp': item.get('timestamp')
                        })
                    data_source = 'supabase'
                else:
                    # Fall back to mock data if result is empty
                    diagnoses = generate_mock_diagnoses(start_date, end_date)
                    data_source = 'mock (empty result)'
            except Exception as db_error:
                print(f"Error retrieving statistics from Supabase: {db_error}")
                # Fall back to mock data
                diagnoses = generate_mock_diagnoses(start_date, end_date)
                data_source = 'mock (database exception)'
        
        # Process data for statistics
        symptom_counts = {}
        disease_counts = {}
        locations = []
        time_series_data = {}
        
        # Prepare time series buckets (days)
        start = datetime.fromisoformat(start_date.replace('Z', ''))
        end = datetime.fromisoformat(end_date.replace('Z', ''))
        days_diff = (end - start).days + 1
        
        # Initialize time series data
        for i in range(days_diff):
            day = (start + timedelta(days=i)).strftime('%Y-%m-%d')
            time_series_data[day] = {'total': 0}
        
        for diagnosis in diagnoses:
            # Count symptoms
            for symptom in diagnosis['symptoms']:
                symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
            
            # Count diseases
            disease = diagnosis['ml_diagnosis']
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
            
            # Aggregate time series data
            try:
                day = datetime.fromisoformat(diagnosis['timestamp'].replace('Z', '')).strftime('%Y-%m-%d')
                if day in time_series_data:
                    time_series_data[day]['total'] += 1
                    
                    # Count by disease
                    if disease not in time_series_data[day]:
                        time_series_data[day][disease] = 0
                    time_series_data[day][disease] += 1
            except (ValueError, TypeError) as e:
                print(f"Error processing timestamp {diagnosis.get('timestamp')}: {e}")
            
            # Collect locations
            if diagnosis.get('latitude') and diagnosis.get('longitude'):
                locations.append({
                    'lat': diagnosis['latitude'],
                    'lng': diagnosis['longitude'],
                    'symptoms': diagnosis['symptoms'],
                    'disease': disease
                })
        
        # Format time series for chart consumption
        formatted_time_series = [
            {
                'date': date,
                'total': data['total'],
                **{k: v for k, v in data.items() if k != 'total'}
            }
            for date, data in time_series_data.items()
        ]
        
        # Get top symptoms and diseases
        top_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        response_data = {
            'symptom_counts': symptom_counts,
            'disease_counts': disease_counts,
            'locations': locations,
            'time_series': formatted_time_series,
            'top_symptoms': top_symptoms,
            'top_diseases': top_diseases,
            'total_diagnoses': len(diagnoses),
            'data_source': data_source
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error in statistics endpoint: {error_msg}")
        return jsonify({'error': error_msg}), 500

def generate_mock_diagnoses(start_date, end_date):
    """Generate mock diagnosis data for demonstration purposes"""
    mock_diagnoses = []
    
    start = datetime.fromisoformat(start_date.replace('Z', ''))
    end = datetime.fromisoformat(end_date.replace('Z', ''))
    days_diff = (end - start).days + 1
    
    # Generate 1-5 diagnoses per day
    for i in range(days_diff):
        day = start + timedelta(days=i)
        for _ in range(random.randint(1, 5)):
            # Random disease
            disease_idx = random.randint(0, len(disease_mapping) - 1)
            disease = disease_mapping[disease_idx]
            
            # Random symptoms (3-5)
            symptoms = random.sample(column_names, min(random.randint(3, 5), len(column_names)))
            
            # Random location around a center point (New York by default)
            base_lat, base_lng = 40.7128, -74.0060
            mock_diagnoses.append({
                'symptoms': symptoms,
                'ml_diagnosis': disease,
                'ml_confidence': random.uniform(60, 95),
                'llm_diagnosis': disease,
                'llm_confidence': random.uniform(55, 90),
                'latitude': base_lat + random.uniform(-0.1, 0.1),
                'longitude': base_lng + random.uniform(-0.1, 0.1),
                'timestamp': day.isoformat()
            })
    
    return mock_diagnoses

if __name__ == "__main__":
    import sys
    
    # Check if there's a command line argument to run in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        print("Running in interactive mode...")
        interactive_test()
    else:
        # Run as web server by default
        print("Starting GeoMed diagnosis API server...")
        app.run(debug=True, host='0.0.0.0', port=5000) 