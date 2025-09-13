## Mail Spam Detector

## Goal

Build a spam mail detector that automatically flags messages as ham (normal) or spam, based solely on their text content.

## Models Tested

1. Baselines (Traditional NLP)

TF-IDF + Logistic Regression → first baseline model.

TF-IDF + Random Forest → more robust tree-based model.

TF-IDF + XGBoost → gradient boosting for stronger performance.

2. Deep Learning

Embedding + CNN → strong results (~98% accuracy).

3. Transfer Learning

BERT

Results were weaker than CNN and DistilBERT.

⚠️ Note: Only trained for 3 epochs due to limited resources.

This short training time explains the low recall and precision for spam detection.

With more epochs and fine-tuning, BERT is expected to perform significantly better.

DistilBERT (a lightweight version of BERT)

Results: ~98.6% accuracy after 6 epochs.

Stronger contextual understanding than TF-IDF.

More efficient than BERT, with better performance in this project.

## Comparative Results

Model	Accuracy
TF-IDF + Logistic Regression	~95%
TF-IDF + Random Forest	~96%
TF-IDF + XGBoost	~97%
CNN (Embedding)	~98%
DistilBERT (Transfer Learning)	~98.6%
BERT (3 epochs, limited run)	~86%

## Conclusion

Classical TF-IDF approaches are solid but lack contextual understanding.

CNN with embeddings achieves very high accuracy.

DistilBERT slightly outperforms CNN and shows the power of Transfer Learning in NLP.

BERT underperformed in this project only because of time/computation limits. With more training, it would likely surpass CNN and DistilBERT.

## Next steps / Future work:

Train BERT and DistilBERT with more epochs.

Experiment with full BERT fine-tuning.


