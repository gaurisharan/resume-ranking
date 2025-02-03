# app.py
from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.secret_key = 'your-secret-key-here'

class ResumeProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF files"""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ' '.join([page.extract_text() for page in reader.pages])
        return text

    def preprocess_text(self, text):
        """Process text with NLP"""
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)

    def calculate_similarity(self, jd_text, resumes):
        """Calculate similarity scores"""
        processed_jd = self.preprocess_text(jd_text)
        processed_resumes = [self.preprocess_text(resume) for resume in resumes]
        
        tfidf_matrix = self.vectorizer.fit_transform([processed_jd] + processed_resumes)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        return cosine_sim[0]

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        # Handle file uploads
        jd_file = request.files['jd']
        resume_files = request.files.getlist('resumes')
        
        # Save files
        jd_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(jd_file.filename))
        jd_file.save(jd_path)
        
        resume_paths = []
        for resume in resume_files:
            path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(resume.filename))
            resume.save(path)
            resume_paths.append(path)
        
        # Process files
        processor = ResumeProcessor()
        
        # Read JD
        with open(jd_path, 'r') as f:
            jd_text = f.read()
        
        # Process resumes
        resume_texts = []
        for path in resume_paths:
            if path.endswith('.pdf'):
                text = processor.extract_text_from_pdf(path)
            else:
                with open(path, 'r') as f:
                    text = f.read()
            resume_texts.append(text)
        
        # Calculate scores
        scores = processor.calculate_similarity(jd_text, resume_texts)
        
        # Prepare results
        results = []
        for path, score in zip(resume_paths, scores):
            results.append({
                'filename': os.path.basename(path),
                'score': round(score, 4)
            })
        
        # Sort results
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Clean up uploaded files
        os.remove(jd_path)
        for path in resume_paths:
            os.remove(path)
        
        return render_template('results.html', results=sorted_results)
    
    return render_template('upload.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
