# Emotion Classification Pipeline
### Year 2 – Block A | Applied Data Science & AI | Breda University of Applied Sciences  
**Author:** [Daria Elena Vlăduțu]

---

## Project Overview  
This project implements an end-to-end Natural Language Processing pipeline for emotion classification from video content. Using traditional NLP methods and transformer models, the pipeline processes multilingual video content to identify and classify emotions expressed in speech, applying concepts from tokenization to state-of-the-art language models.

---

## Objectives  
- Build a complete NLP pipeline from video to emotion classification
- Implement speech-to-text transcription capabilities  
- Develop multilingual translation models  
- Create robust emotion classification using various NLP techniques
- Apply transformer architectures for improved performance
- Gain hands-on experience with HuggingFace models and PyTorch

---

## Pipeline Architecture  

### Core Components  
1. **Speech Transcription**: Convert video audio to text using ASR models
2. **Translation Module**: Translate non-English content to English 
3. **Emotion Tagging**: Classify emotions (6 core + neutral) per sentence
4. **Output Generation**: Structured file with transcriptions, translations, and emotion labels

### Medal Challenges (Optional Enhancements)
- **Bronze**: Implement transcription error correction
- **Silver**: Add emotion intensity classification (mild/moderate/strong)  
- **Gold**: Enable multi-label emotion prediction

### Technical Implementation
- N-gram language models for text analysis
- Logistic regression & Naive Bayes for baseline classification
- Word embeddings (Word2Vec) for semantic representation
- RNNs/LSTMs for sequential processing
- Transformer models for state-of-the-art performance
- Fine-tuning pre-trained models from HuggingFace

---

## Key Results  
- Successfully processed [X] hours of video content
- Achieved [X]% accuracy on emotion classification
- Reduced translation errors by [X]% through custom fine-tuning
- Implemented real-time processing capability
- Created modular, reusable pipeline components

---

## Skills Demonstrated  
- Natural Language Processing fundamentals
- Deep learning for NLP (RNNs, Transformers)
- Speech recognition & machine translation
- Emotion/sentiment analysis
- Model fine-tuning & transfer learning
- Pipeline development & integration
- Performance optimization & evaluation

---

## Tools & Technologies  
- **Languages**: Python
- **Deep Learning**: PyTorch, TensorFlow/Keras
- **NLP Libraries**: HuggingFace Transformers, NLTK, spaCy
- **Models**: BERT, GPT, Whisper (ASR), MarianMT (translation)
- **Visualization**: Matplotlib, Seaborn
- **Project Management**: Trello/Microsoft Planner
- **Version Control**: Git & GitHub

---

## Repository Structure
```
block-a-nlp-emotion/
├── notebooks/                   # Jupyter notebooks for experiments
│   ├── embeddings.ipynb
│   ├── svm.ipynb
│   └── rnns.ipynb
├── src/                         # Source code modules
│   ├── transcription/          # ASR components
│   ├── translation/            # Translation models
│   └── classification/         # Emotion classifiers
├── data/                        # Sample data and datasets
├── models/                      # Saved model weights
├── results/                     # Output files and evaluations
├── docs/                        # Documentation and reports
│   ├── presentation.pdf
│   └── learning_log.md
└── README.md
```

---

## Installation & Usage

### Prerequisites
```bash
pip install torch transformers nltk spacy
pip install whisper huggingface-hub datasets
```

### Quick Start
```python
from pipeline import EmotionPipeline

# Initialize pipeline
pipeline = EmotionPipeline(
    transcribe=True,
    translate=True,
    classify_emotions=True
)

# Process video
results = pipeline.process_video("input_video.mp4")

# Access results
print(results['transcription'])
print(results['emotions'])
```

---

## Deliverables  
- Learning Log (GitHub Wiki) - Individual assessment
- Work Log - Individual tracking document
- Peer Feedback Reports (Weeks 5 & 8)
- Team Presentation (~15 minutes)
- Complete Python implementation with documentation
- Medal challenge implementations (if applicable)

---

## References
- Jurafsky & Martin (2025). *Speech and Language Processing*, 3rd edition
- Bird, Klein, & Loper. *Natural Language Processing with Python*
- HuggingFace Transformers documentation
- Course materials from BUas ADS&AI Block A

---

## Contact
For questions about this project, please contact:
- **Mentor**: [Mentor Name] - [email]
- **Block Responsible**: Myrthe Buckens - buckens.m@buas.nl