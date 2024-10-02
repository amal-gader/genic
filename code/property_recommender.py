import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from data_loaders import load_dataset_processed

from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, precision_score, recall_score



def recomm_data(df):
    entity_property_matrix = df.drop(columns=['head', 'head types', 'head_description', 'relation', 'text']).to_numpy()
    x_train, x_test = train_test_split(df['text'], test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(entity_property_matrix, test_size=0.2, random_state=42 )
    y_test_removed = np.zeros_like(entity_property_matrix[x_test.index])
    # Find the first non-zero property for each entity in x_test
    first_property_indices = np.argmax(entity_property_matrix[x_test.index], axis=1)
    # Create a new array with zeros and set the first property to 1
    for i, idx in enumerate(first_property_indices):
        y_test_removed[i, idx] = 1
    entity_property_matrix[x_test.index] = y_test_removed
    return entity_property_matrix, x_train, x_test, y_train, y_test


class CollaborativeFiltering:
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.knn_model = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine')
 
    def fit(self, entity_property_matrix):
        self.knn_model.fit(entity_property_matrix)

    def predict(self,entity_property_matrix, entity_idx):
        distances, indices = self.knn_model.kneighbors(entity_property_matrix[entity_idx].reshape(1, -1))
        similar_indices = indices.flatten()
        similar_items = entity_property_matrix[similar_indices]
        return np.mean(similar_items, axis=0)

# Create a Content-Based Filtering model using TF-IDF
class ContentBasedFiltering:
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.vectorizer = TfidfVectorizer()

    def fit(self, x_train):
        self.tfidf_matrix = self.vectorizer.fit_transform(x_train)

    def predict(self, entity_text):
        entity_vector = self.vectorizer.transform([entity_text])
        cosine_similarities = cosine_similarity(entity_vector, self.tfidf_matrix).flatten()
        similar_indices = np.argsort(cosine_similarities)[-self.n_neighbors:]
        similar_items = y_train[similar_indices]
        return   np.mean(similar_items, axis=0)
    
# Hybrid Model that combines CF and CBF
class HybridModel:
    def __init__(self, alpha=1):
        self.cf_model = CollaborativeFiltering()
        self.cbf_model = ContentBasedFiltering()
        self.alpha = alpha  # Weight between CF and CBF

    def fit(self, x_train, entity_property_matrix):
        self.cf_model.fit(entity_property_matrix)
        self.cbf_model.fit(x_train)

    def predict(self, entity_property_matrix,x_test, entity_idx):
        # CF Prediction
        cf_prediction = self.cf_model.predict(entity_property_matrix, entity_idx)
        # CBF Prediction
        cbf_prediction = self.cbf_model.predict( x_test[entity_idx])
        combined_prediction = (self.alpha * cf_prediction) + ((1 - self.alpha) * cbf_prediction)
        combined_prediction = normalize(combined_prediction.reshape(1, -1)).flatten()
        
        return combined_prediction
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, choices=['codex', 'fb15k', 'wn18rr'], help='The dataset to be used: codex, fb15k, or wn18rr.')
    args = parser.parse_args()
    df = load_dataset_processed(args.dataset)
    
    
    entity_property_matrix, x_train, x_test, y_train, y_test = recomm_data(df)
    hybrid_model = HybridModel(alpha=0.5)
    hybrid_model.fit(x_train, entity_property_matrix)
    predictions = []
    for i in x_test.index:
        prediction = hybrid_model.predict(entity_property_matrix, x_test, i)
        predictions.append(prediction)
        
    # Binarize predictions
    binary_predictions = []
    for prediction in predictions:
        binary_predictions.append((prediction > 0.2).astype(int))

    # Evaluate
    f1 = f1_score(y_test, binary_predictions, average='macro')
    precision = precision_score(y_test, binary_predictions, average='macro')
    recall = recall_score(y_test, binary_predictions, average='macro')

    print(f'F1 Score: {f1}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

        