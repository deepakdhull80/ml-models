import numpy as np

'''
Assume:
Document: "free cash win"
Class: "spam"
Word "cash" never appeared in spam training documents.
Then:
P("free"∣spam)⋅P("cash"∣spam)⋅P("win"∣spam)=x⋅0⋅z=0
Even if "free" and "win" are strong spam indicators, the zero wipes out everything.
'''

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = self.classes.shape[0]
        n_samples, n_features = X.shape
        # calculate the log prior
        self.log_prior = np.log(np.bincount(y)/y.shape[0])
        # calculate the log conditional prob
        self.log_cond_prob_ = np.zeros((n_classes, n_features))
        for c in range(n_classes):
            idx = y == c
            x_ = X[idx]
            self.log_cond_prob_[c] = np.log((x_.sum(axis=0) + self.alpha)/ (x_.sum(axis=0).sum() + self.alpha + n_features))
    
    def predict(self, X):
        log_prob = np.dot(X, self.log_cond_prob_.T) + self.log_prior
        return self.classes[np.argmax(log_prob, axis=1)]



if __name__ == "__main__":
    documents = [
        "buy cheap meds",
        "cheap meds online",
        "meeting tomorrow",
        "project deadline"
    ]

    labels = np.array([1, 1, 0, 0])
    vocabs = set([word for doc in documents for word in doc.split()])
    vocab_idx = { v:i for i, v in enumerate(vocabs) }
    
    # vectorize
    X = np.zeros((len(documents), len(vocabs)))
    for i, doc in enumerate(documents):
        for token in doc.split():
            X[i, vocab_idx[token]] += 1
    
    
    model = NaiveBayes()
    
    model.fit(X, labels)
    print(model.predict(X))