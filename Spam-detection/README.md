## Mail Spam Detection

# Goal

The objective of this project is to build a spam detector that automatically flags spam messages based solely on the mails text content.
 
# Dataset

•	Source: SMS Spam Collection Dataset 
•	Size: 5,572 messages (≈ 87% ham, 13% spam)
•	Challenge: class imbalance (ham >> spam)
 
# Preprocessing Steps

•	Text cleaning (lowercasing, punctuation removal, stopwords removal)
•	Tokenization & Lemmatization
•	Train/test split (80/20)
 
# Models & Results

 1. Baseline: TF-IDF + Logistic Regression
•	Vectorized SMS with TF-IDF
•	Trained a Logistic Regression classifier
•	Result: High accuracy with strong spam recall → strong baseline

 2. TF-IDF + Random Forest / XGBoost
•	Used same TF-IDF features with Random Forest and XGBoost
•	Captured more complex interactions
•	Result: Similar or slightly better accuracy than LR, but higher training cost

 3. Embedding + CNN
•	Represented text with trainable embeddings (20,000 vocab, 100 tokens max)
•	Applied Conv1D + GlobalMaxPooling + Dense layers
•	Result:
o	Accuracy ≈ 98%
o	Very strong on spam recall → CNN captured local n-gram features

 # Conclusion: 
 
 CNN + Embedding is the best-performing model under our constraints.
 TF-IDF + LR remains the most interpretable and efficient baseline.
 
# Future Work / Limitations

We also experimented with BERT (transfer learning) but results were limited:

•	Only trained 3 epochs due to time/compute limits → low spam recall
•	Large models like BERT need longer fine-tuning and class balancing to work well on small, imbalanced datasets

# Next Steps to Improve BERT

•	Train longer (5–10 epochs) with early stopping
•	Use DistilBERT (lighter, faster)
•	Collect or augment more spam data
 
# Key Takeaways

•	Traditional ML (TF-IDF + LR) is strong and interpretable

•	Deep learning (Embedding + CNN) achieves the highest accuracy

•	Transfer learning (BERT) shows potential, but requires more resources to outperform lighter models
