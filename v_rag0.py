import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

class DiseaseRAGSystem:
    def __init__(self, json_file_path, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the RAG system with disease data and embedding model.
        
        Args:
            json_file_path: Path to JSON file containing disease data
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.diseases_data = self._load_data(json_file_path)
        self.symptom_embeddings = self._precompute_symptom_embeddings()
        
    def _load_data(self, json_file_path):
        """Load disease data from JSON file."""
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def _precompute_symptom_embeddings(self):
        """Precompute embeddings for all official symptoms."""
        symptom_embeddings = {}
        
        for disease_info in self.diseases_data:
            disease = disease_info['disease']
            symptoms = disease_info['symptoms']
            
            # Compute embeddings for each symptom
            embeddings = self.model.encode(symptoms)
            symptom_embeddings[disease] = {
                'symptoms': symptoms,
                'embeddings': embeddings
            }
        
        return symptom_embeddings
    
    def extract_symptoms_from_query(self, query, similarity_threshold=0.5):
        """
        Extract symptoms from user query using semantic similarity.
        
        Args:
            query: User's natural language query
            similarity_threshold: Minimum cosine similarity to consider a match
            
        Returns:
            List of tuples (matched_symptom, similarity_score, disease)
        """
        # Encode the query
        query_embedding = self.model.encode([query])[0]
        
        matched_symptoms = []
        
        # Compare query with all official symptoms
        for disease, symptom_data in self.symptom_embeddings.items():
            symptoms = symptom_data['symptoms']
            embeddings = symptom_data['embeddings']
            
            # Calculate cosine similarity between query and each symptom
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            
            # Find matches above threshold
            for symptom, similarity in zip(symptoms, similarities):
                if similarity >= similarity_threshold:
                    matched_symptoms.append((symptom, similarity, disease))
        
        # Sort by similarity score (highest first)
        matched_symptoms.sort(key=lambda x: x[1], reverse=True)
        
        return matched_symptoms
    
    def extract_symptoms_by_sentence(self, query, similarity_threshold=0.45):
        """
        Extract symptoms by splitting query into sentences and matching each.
        This provides better granularity for multi-symptom queries.
        
        Args:
            query: User's natural language query
            similarity_threshold: Minimum cosine similarity to consider a match
            
        Returns:
            List of unique matched symptoms with their diseases
        """
        # Split query into sentences
        sentences = re.split(r'[.!?]+', query)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If no sentences found, treat entire query as one sentence
        if not sentences:
            sentences = [query]
        
        # Encode all sentences
        sentence_embeddings = self.model.encode(sentences)
        
        matched_symptoms = {}  # Use dict to avoid duplicates
        
        # Compare each sentence with all official symptoms
        for sentence, sent_embedding in zip(sentences, sentence_embeddings):
            for disease, symptom_data in self.symptom_embeddings.items():
                symptoms = symptom_data['symptoms']
                embeddings = symptom_data['embeddings']
                
                # Calculate cosine similarity
                similarities = cosine_similarity([sent_embedding], embeddings)[0]
                
                # Find matches above threshold
                for symptom, similarity in zip(symptoms, similarities):
                    if similarity >= similarity_threshold:
                        # Keep only the highest similarity for each symptom
                        key = (symptom, disease)
                        if key not in matched_symptoms or matched_symptoms[key] < similarity:
                            matched_symptoms[key] = similarity
        
        # Convert to list format
        result = [(symptom, score, disease) for (symptom, disease), score in matched_symptoms.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result
    
    def calculate_disease_scores(self, matched_symptoms):
        """
        Calculate scores for each disease based on matched symptoms.
        
        Scoring formula: sum of (1 / num_symptoms_in_disease / num_matched_symptoms)
        for each matched symptom.
        
        Args:
            matched_symptoms: List of tuples (symptom, similarity, disease)
            
        Returns:
            Dictionary of disease scores sorted by score (highest first)
        """
        # Group matched symptoms by disease
        disease_matches = {}
        for symptom, similarity, disease in matched_symptoms:
            if disease not in disease_matches:
                disease_matches[disease] = []
            disease_matches[disease].append((symptom, similarity))
        
        # Calculate scores
        disease_scores = {}
        total_matched_symptoms = len(set([s[0] for s in matched_symptoms]))
        
        for disease, matches in disease_matches.items():
            # Get total number of symptoms for this disease
            num_disease_symptoms = len(self.symptom_embeddings[disease]['symptoms'])
            
            # Calculate score for each matched symptom
            score = 0
            for symptom, similarity in matches:
                # Base score according to your formula
                base_score = (1.0 / num_disease_symptoms) + (1.0 / total_matched_symptoms)
                # Weight by similarity score
                score += base_score * similarity
            
            disease_scores[disease] = {
                'score': score,
                'matched_symptoms': [s[0] for s in matches],
                'num_matches': len(matches),
                'total_symptoms': num_disease_symptoms
            }
        
        # Sort by score
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        return dict(sorted_diseases)
    
    def get_disease_info(self, disease_name):
        """Get full information for a specific disease."""
        for disease_info in self.diseases_data:
            if disease_info['disease'] == disease_name:
                return disease_info
        return None
    
    def diagnose(self, query, top_k=3, similarity_threshold=0.45):
        """
        Main method to diagnose disease from user query.
        
        Args:
            query: User's natural language query describing symptoms
            top_k: Number of top diseases to return
            similarity_threshold: Minimum similarity for symptom matching
            
        Returns:
            Dictionary containing diagnosis results
        """
        # Extract symptoms using sentence-based matching
        matched_symptoms = self.extract_symptoms_by_sentence(query, similarity_threshold)
        
        if not matched_symptoms:
            return {
                'status': 'no_match',
                'message': 'Could not identify any symptoms from the query. Please describe your symptoms more clearly.',
                'matched_symptoms': [],
                'top_diseases': []
            }
        
        # Calculate disease scores
        disease_scores = self.calculate_disease_scores(matched_symptoms)
        
        # Get top K diseases
        top_diseases = []
        for i, (disease, score_info) in enumerate(disease_scores.items()):
            if i >= top_k:
                break
            
            disease_info = self.get_disease_info(disease)
            top_diseases.append({
                'disease': disease,
                'score': score_info['score'],
                'matched_symptoms': score_info['matched_symptoms'],
                'num_matches': score_info['num_matches'],
                'total_symptoms': score_info['total_symptoms'],
                'all_symptoms': disease_info['symptoms'],
                'precautions': disease_info['precautions']
            })
        
        return {
            'status': 'success',
            'query': query,
            'matched_symptoms': list(set([s[0] for s in matched_symptoms])),
            'top_diseases': top_diseases,
            'best_match': top_diseases[0] if top_diseases else None
        }
    
    def generate_response(self, diagnosis_result):
        """
        Generate a natural language response based on diagnosis.
        
        Args:
            diagnosis_result: Output from diagnose() method
            
        Returns:
            Formatted string response
        """
        if diagnosis_result['status'] == 'no_match':
            return diagnosis_result['message']
        
        best_match = diagnosis_result['best_match']
        
        response = f"Based on your symptoms, you may have **{best_match['disease']}**.\n\n"
        response += f"**Matched symptoms ({best_match['num_matches']}/{best_match['total_symptoms']}):**\n"
        for symptom in best_match['matched_symptoms']:
            response += f"- {symptom}\n"
        
        response += f"\n**Recommended precautions:**\n"
        for precaution in best_match['precautions']:
            response += f"- {precaution}\n"
        
        if len(diagnosis_result['top_diseases']) > 1:
            response += f"\n**Other possible conditions:**\n"
            for disease in diagnosis_result['top_diseases'][1:]:
                response += f"- {disease['disease']} (score: {disease['score']:.3f})\n"
        
        response += "\n*Note: This is an automated assessment. Please consult a healthcare professional for proper diagnosis and treatment.*"
        
        return response


# Example usage
if __name__ == "__main__":
    # Initialize the system
    print("Initializing Disease RAG System...")
    rag = DiseaseRAGSystem('medical_dataset.json')
    
    # Example query
    query = "I have rashes on my shoulder which itches a lot. There are also dark patches on my neck region."
    print("Please describe your issue in detail: ")
    #input(query)

    print(f"\nQuery: {query}\n")
    print("Diagnosing...\n")
    
    # Get diagnosis
    result = rag.diagnose(query, top_k=3, similarity_threshold=0.45)
    
    # Generate and print response
    response = rag.generate_response(result)
    print(response)
    
    # Print detailed results
    print("\n" + "="*50)
    print("DETAILED RESULTS")
    print("="*50)
    print(f"\nMatched symptoms from query: {result['matched_symptoms']}")
    print(f"\nTop diseases:")
    for disease in result['top_diseases']:
        print(f"\n{disease['disease']}:")
        print(f"  Score: {disease['score']:.4f}")
        print(f"  Matched: {disease['num_matches']}/{disease['total_symptoms']} symptoms")
        print(f"  Matched symptoms: {disease['matched_symptoms']}")
