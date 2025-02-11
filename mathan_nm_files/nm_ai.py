import spacy
import pandas as pd
import difflib
import warnings
import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning

# Load spaCy NLP model
nlp = spacy.load('en_core_web_sm')

# Ignore convergence warnings
warnings.simplefilter("ignore", ConvergenceWarning)

# Function to extract noun-modifiers using spaCy
def extract_noun_modifiers(text):
    doc = nlp(text)
    modifiers = [token.text for token in doc if token.dep_ in ("amod", "compound") and token.head.pos_ == "NOUN"]
    return ' '.join(modifiers) if modifiers else ""

# Function to read Excel files safely
def read_excel_file(file_path):
    try:
        df = pd.read_excel(file_path)
        required_columns = ['Material Source', 'Noun', 'Modifier']
        if all(col in df.columns for col in required_columns):
            return df[required_columns].dropna()
        else:
            print(f"Error: {file_path} does not contain required columns.")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Function to find best match for Material Source
def find_best_match(source, trained_dict):
    keys = list(trained_dict.keys())
    closest_matches = difflib.get_close_matches(source.lower(), [key.lower() for key in keys], n=1, cutoff=0.5)
    if closest_matches:
        matched_key = next((key for key in keys if key.lower() == closest_matches[0]), None)
        return trained_dict.get(matched_key, ("", ""))
    return "", ""

# Function to fetch material information from the web
def fetch_material_info(material_name):
    try:
        query = f"{material_name} material meaning"
        url = f"https://www.google.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        result = soup.find("div", class_="BNeawe").text
        return result if result else ""
    except Exception:
        return ""

# Function to train ML models
def train_model(training_data):
    le_source = LabelEncoder()
    le_noun = LabelEncoder()
    le_modifier = LabelEncoder()
    
    training_data['Material Source Encoded'] = le_source.fit_transform(training_data['Material Source'].astype(str))
    training_data['Noun Encoded'] = le_noun.fit_transform(training_data['Noun'].astype(str))
    training_data['Modifier Encoded'] = le_modifier.fit_transform(training_data['Modifier'].astype(str))
    
    X = training_data[['Material Source Encoded']]
    y_noun = training_data['Noun Encoded']
    y_modifier = training_data['Modifier Encoded']
    
    X_train, X_test, y_train_noun, y_test_noun = train_test_split(X, y_noun, test_size=0.2, random_state=42)
    X_train, X_test, y_train_modifier, y_test_modifier = train_test_split(X, y_modifier, test_size=0.2, random_state=42)
    
    model_noun = LogisticRegression(max_iter=5000, solver='saga')
    model_noun.fit(X_train, y_train_noun)
    
    model_modifier = LogisticRegression(max_iter=5000, solver='saga')
    model_modifier.fit(X_train, y_train_modifier)
    
    return model_noun, model_modifier, le_source, le_noun, le_modifier

# Function to predict noun and modifier
def predict_noun_modifier(model_noun, model_modifier, le_source, le_noun, le_modifier, material_sources, trained_dict):
    predictions = []
    
    for source in material_sources:
        noun, modifier = find_best_match(source, trained_dict)
        
        if not noun or not modifier:
            try:
                encoded_source = le_source.transform([source])
                X_input = pd.DataFrame({'Material Source Encoded': encoded_source})
                predicted_noun = model_noun.predict(X_input)[0]
                predicted_modifier = model_modifier.predict(X_input)[0]
                noun = le_noun.inverse_transform([predicted_noun])[0]
                modifier = le_modifier.inverse_transform([predicted_modifier])[0]
            except ValueError:
                noun = extract_noun_modifiers(source)
                modifier = fetch_material_info(source)
        
        # Ensure no "Unknown" values
        if not noun:
            noun = fetch_material_info(source)
        if not modifier:
            modifier = extract_noun_modifiers(source)
        
        predictions.append({
            'Material Source': source,
            'Noun': noun,
            'Modifier': modifier
        })
    
    return predictions

# Function to export predictions to Excel
def export_to_excel(data, output_path):
    try:
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)
        print(f"✅ Data has been exported to {output_path}")
    except Exception as e:
        print(f"❌ Error exporting data: {e}")

# Main function to execute the workflow
def main(training_files, input_file, output_file):
    combined_data = []
    trained_dict = {}
    
    for file in training_files:
        data = read_excel_file(file)
        if data is not None:
            combined_data.append(data)
            trained_dict.update({row['Material Source']: (row['Noun'], row['Modifier']) for _, row in data.iterrows()})
    
    if combined_data:
        training_data = pd.concat(combined_data, ignore_index=True)
        model_noun, model_modifier, le_source, le_noun, le_modifier = train_model(training_data)
        
        input_data = pd.read_excel(input_file)
        if 'Material Source' not in input_data.columns:
            print("❌ Input file does not contain 'Material Source' column.")
            return

        material_sources = input_data['Material Source'].astype(str).tolist()
        
        predictions = predict_noun_modifier(model_noun, model_modifier, le_source, le_noun, le_modifier, material_sources, trained_dict)
        
        export_to_excel(predictions, output_file)
    else:
        print("❌ No valid training data found.")


# Example usage
training_files = ['E:/NM1.xlsx', 'E:/NM2.xlsx', 'E:/NM3.xlsx']
input_file = 'E:/SRC1.xlsx'  
output_file = 'E:/machine learning/noun_modifier_updated.xlsx'  

# Run the process
main(training_files, input_file, output_file)
