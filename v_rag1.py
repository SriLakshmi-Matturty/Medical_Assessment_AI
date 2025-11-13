import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
from rank_bm25 import BM25Okapi
import re

class DiseaseRAGSystem:
    def __init__(self, json_file_path, model_name='all-MiniLM-L6-v2', alpha=0.7):
        """
        Initialize the RAG system with both dense (vector) and sparse (BM25) retrieval.

        Args:
            json_file_path: Path to JSON file containing disease data
            model_name: Name of the sentence transformer model
            alpha: Weight for dense vs sparse retrieval (0.7 = 70% dense, 30% sparse)
        """
        self.model = SentenceTransformer(model_name)
        self.diseases_data = self._load_data(json_file_path)
        self.symptom_embeddings = self._precompute_symptom_embeddings()
        self.alpha = alpha  # weight for fusion

        # Prepare corpus for BM25
        self.symptom_texts = []
        for d in self.diseases_data:
            for s in d["symptoms"]:
                self.symptom_texts.append((s, d["disease"]))  # keep mapping

        tokenized_corpus = [s[0].lower().split() for s in self.symptom_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    # --------------------- Data Loading ---------------------
    def _load_data(self, json_file_path):
        with open(json_file_path, 'r') as f:
            return json.load(f)

    def _precompute_symptom_embeddings(self):
        """Precompute embeddings for all symptoms (for dense retrieval)."""
        symptom_embeddings = {}
        for disease_info in self.diseases_data:
            disease = disease_info['disease']
            symptoms = disease_info['symptoms']
            embeddings = self.model.encode(symptoms, show_progress_bar=False)
            symptom_embeddings[disease] = {
                'symptoms': symptoms,
                'embeddings': embeddings
            }
        return symptom_embeddings

    # --------------------- Hybrid Matching ---------------------
    def _get_hybrid_scores(self, query):
        """
        Combine dense and BM25 (sparse) retrieval scores for each symptom.
        """
        query_embedding = self.model.encode([query])[0]

        # --- Dense similarities ---
        all_symptoms = [s[0] for s in self.symptom_texts]
        dense_embeddings = np.vstack(
            [self.symptom_embeddings[s[1]]['embeddings']
             [self.symptom_embeddings[s[1]]['symptoms'].index(s[0])]
             for s in self.symptom_texts]
        )
        dense_scores = cosine_similarity([query_embedding], dense_embeddings)[0]

        # --- BM25 scores ---
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)

        # --- Normalize and fuse ---
        dense_norm = minmax_scale(dense_scores)
        bm25_norm = minmax_scale(bm25_scores)
        final_scores = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm

        return list(zip(all_symptoms, final_scores, [s[1] for s in self.symptom_texts]))

    def extract_symptoms_by_sentence(self, query, similarity_threshold=0.45):
        """
        Extract symptoms by combining dense + BM25 (hybrid retrieval).
        """
        sentences = re.split(r'[.!?]+', query)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            sentences = [query]

        matched_symptoms = {}

        for sentence in sentences:
            hybrid_scores = self._get_hybrid_scores(sentence)

            for symptom, score, disease in hybrid_scores:
                if score >= similarity_threshold:
                    key = (symptom, disease)
                    if key not in matched_symptoms or matched_symptoms[key] < score:
                        matched_symptoms[key] = score

        result = [(symptom, score, disease) for (symptom, disease), score in matched_symptoms.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    # --------------------- Disease Scoring ---------------------
    def calculate_disease_scores(self, matched_symptoms):
        disease_matches = {}
        for symptom, similarity, disease in matched_symptoms:
            disease_matches.setdefault(disease, []).append((symptom, similarity))

        disease_scores = {}
        total_matched = len(set(s for s, _, _ in matched_symptoms))

        for disease, matches in disease_matches.items():
            num_symptoms = len(self.symptom_embeddings[disease]['symptoms'])
            score = sum(((1 / num_symptoms) + (1 / total_matched)) * sim for _, sim in matches)
            disease_scores[disease] = {
                'score': score,
                'matched_symptoms': [s[0] for s in matches],
                'num_matches': len(matches),
                'total_symptoms': num_symptoms
            }

        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        return dict(sorted_diseases)

    # --------------------- Diagnosis ---------------------
    def get_disease_info(self, disease_name):
        return next((d for d in self.diseases_data if d['disease'] == disease_name), None)

    def diagnose(self, query, top_k=3, similarity_threshold=0.45):
        matched_symptoms = self.extract_symptoms_by_sentence(query, similarity_threshold)

        if not matched_symptoms:
            return {
                'status': 'no_match',
                'message': 'No symptoms recognized from the query. Try rephrasing.',
                'matched_symptoms': [],
                'top_diseases': []
            }

        disease_scores = self.calculate_disease_scores(matched_symptoms)
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

    # --------------------- Response Generation ---------------------
    def generate_response(self, diagnosis_result):
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

        response += "\n*Note: This is an automated assessment. Please consult a healthcare professional for confirmation.*"
        return response


# --------------------- Example Run ---------------------
if __name__ == "__main__":
    print("Initializing Hybrid Disease RAG System...")
    rag = DiseaseRAGSystem('medical_dataset.json', alpha=0.7)

    query = "I have rashes on my shoulder which itches a lot. There are also dark patches on my neck region."
    print(f"\nQuery: {query}\n")

    print("Diagnosing...\n")
    result = rag.diagnose(query, top_k=3, similarity_threshold=0.45)

    response = rag.generate_response(result)
    print(response)

    print("\n" + "="*50)
    print("DETAILED RESULTS")
    print("="*50)
    print(f"\nMatched symptoms: {result['matched_symptoms']}")
    for disease in result['top_diseases']:
        print(f"\n{disease['disease']}:")
        print(f"  Score: {disease['score']:.4f}")
        print(f"  Matched: {disease['num_matches']}/{disease['total_symptoms']}")
        print(f"  Matched symptoms: {disease['matched_symptoms']}")
