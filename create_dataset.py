import pandas as pd
import json

# --- Configuration ---
# Set the paths to your CSV files
symptoms_file_path = 'DiseaseAndSymptoms.csv'
precautions_file_path = 'Disease precaution.csv'
output_json_path = 'medical_dataset.json'

# Set the name of the column that contains the disease name in both files
# This is case-sensitive!
disease_column_name = 'Disease' 

# --- Main Script ---

def process_data():
    """
    Reads symptoms and precautions CSVs, merges them,
    and creates a structured JSON file.
    """
    
    # This dictionary will store all our aggregated data.
    # Its structure will be:
    # {
    #   "Gastroenteritis": {"symptoms": set(), "precautions": []},
    #   "Influenza": {"symptoms": set(), "precautions": []}
    # }
    disease_data = {}

    # --- Step 1: Process the Symptoms CSV ---
    print(f"Processing '{symptoms_file_path}'...")
    try:
        df_symptoms = pd.read_csv(symptoms_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{symptoms_file_path}'")
        return
    except Exception as e:
        print(f"Error reading symptoms CSV: {e}")
        return

    # Get a list of all column names *except* the disease column
    # These are assumed to be symptom columns.
    symptom_columns = df_symptoms.columns.drop(disease_column_name)

    for index, row in df_symptoms.iterrows():
        disease_name = row[disease_column_name]
        
        # Skip rows that don't have a disease name
        if pd.isna(disease_name):
            continue
        
        disease_name = disease_name.strip()

        # If this is the first time we see this disease, add it to our dictionary
        if disease_name not in disease_data:
            disease_data[disease_name] = {"symptoms": set(), "precautions": []}
        
        # Iterate through all possible symptom columns for this row
        for col in symptom_columns:
            symptom = row[col]
            
            # If the symptom is not 'nan' (empty), add it to our set
            # Using a set automatically handles duplicate symptoms
            if pd.notna(symptom):
               cleaned_symptom = str(symptom).strip().replace('_', ' ')
               disease_data[disease_name]["symptoms"].add(cleaned_symptom)

    print(f"Found and processed {len(disease_data)} unique diseases from symptoms file.")

    # --- Step 2: Process the Precautions CSV ---
    print(f"Processing '{precautions_file_path}'...")
    try:
        df_precautions = pd.read_csv(precautions_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{precautions_file_path}'")
        return
    except Exception as e:
        print(f"Error reading precautions CSV: {e}")
        return

    # Get a list of all column names *except* the disease column
    precaution_columns = df_precautions.columns.drop(disease_column_name)

    for index, row in df_precautions.iterrows():
        disease_name = row[disease_column_name]
        
        if pd.isna(disease_name):
            continue
            
        disease_name = disease_name.strip()

        # Find the matching disease in our dictionary and add the precautions
        if disease_name in disease_data:
            # Iterate through all possible precaution columns for this row
            for col in precaution_columns:
                precaution = row[col]
                
                # If the precaution is not 'nan', add it to the list
                if pd.notna(precaution):
                    disease_data[disease_name]["precautions"].append(str(precaution).strip())
        else:
            print(f"Warning: Disease '{disease_name}' from precautions file"
                  " was not found in the symptoms file. It will be skipped.")

    # --- Step 3: Convert to the final JSON list structure ---
    print("Merging and formatting data...")
    final_json_list = []

    for disease_name, data in disease_data.items():
        
        # Convert the set of symptoms to a list for JSON serialization
        symptoms_list = list(data["symptoms"])
        
        final_json_list.append({
            "disease": disease_name,
            "symptoms": symptoms_list,
            "precautions": data["precautions"]
        })

    # --- Step 4: Write the output JSON file ---
    try:
        with open(output_json_path, 'w') as f:
            json.dump(final_json_list, f, indent=4)
        print(f"Successfully created '{output_json_path}'!")
    except Exception as e:
        print(f"Error writing JSON file: {e}")

# --- Run the script ---
if __name__ == "__main__":
    process_data()
