import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import pandas as pd

class ResumeProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def extract_text_from_pdf(self, file):
        """Extract text from PDF file object"""
        reader = PyPDF2.PdfReader(file)
        return ' '.join([page.extract_text() for page in reader.pages])

    def preprocess_text(self, text):
        """Process text with NLP pipeline"""
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)

    def calculate_similarity(self, jd_text, resumes):
        """Calculate TF-IDF cosine similarity scores"""
        processed_jd = self.preprocess_text(jd_text)
        processed_resumes = [self.preprocess_text(resume) for resume in resumes]
        
        tfidf_matrix = self.vectorizer.fit_transform([processed_jd] + processed_resumes)
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

def main():
    st.set_page_config(page_title="Resume Ranker", layout="wide")
    st.title("📄 AI-Powered Resume Ranking System")
    
    processor = ResumeProcessor()
    
    with st.sidebar:
        st.header("Upload Files")
        jd_file = st.file_uploader("Job Description (TXT)", type="txt")
        resume_files = st.file_uploader("Resumes (PDF/TXT)", 
                                      type=["pdf", "txt"],
                                      accept_multiple_files=True)
    
    if jd_file and resume_files:
        try:
            # Process job description
            jd_text = jd_file.read().decode()
            
            # Process resumes
            resume_texts = []
            for file in resume_files:
                if file.type == "application/pdf":
                    text = processor.extract_text_from_pdf(file)
                else:
                    text = file.read().decode()
                resume_texts.append(text)
            
            # Calculate scores
            scores = processor.calculate_similarity(jd_text, resume_texts)
            
            # Create results dataframe
            results = pd.DataFrame({
                "Filename": [f.name for f in resume_files],
                "Score": scores
            }).sort_values("Score", ascending=False)
            
            # Display results
            st.subheader("Ranking Results")
            st.dataframe(
                results,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        format="%.4f",
                        min_value=0,
                        max_value=1.0
                    )
                },
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")

if __name__ == "__main__":
    main()
