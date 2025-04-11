import os
import google.generativeai as genai

def test_llm():
    # API key provided directly
    API_KEY = "AIzaSyBEH6r9XqCM0yaYeUINeF2dc_rawjPVhNA"
    
    try:
        # Configure the API
        genai.configure(api_key=API_KEY)
        print("Successfully configured Google GenerativeAI")
        
        # Create the model with correct name from the list
        model = genai.GenerativeModel('gemini-1.5-pro')
        print("Successfully loaded Gemini 1.5 Pro model")

        # Test prompt
        prompt = """As a medical AI assistant, analyze these symptoms: high fever, cough, fatigue

What is the most likely diagnosis? Consider these possible conditions: Common Cold, Influenza, COVID-19

Please provide:
1. The most likely diagnosis
2. Your confidence level (0-100%)
3. Brief explanation for your diagnosis

Format your response as JSON:
{
  "diagnosis": "disease name",
  "confidence": confidence_percentage,
  "explanation": "your reasoning"
}
"""
        # Get the response
        response = model.generate_content(prompt)

        print("\nGemini Pro Response:")
        print(response.text)

    except Exception as e:
        print(f"Error: {e}")
        print("\nIf you're seeing authentication errors, please:")
        print("1. Go to https://makersuite.google.com/app/apikey")
        print("2. Create an API key")
        print("3. Set it using $env:GOOGLE_API_KEY='your-key-here' in PowerShell")

if __name__ == "__main__":
    test_llm() 