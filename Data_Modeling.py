# Analysis imports
import numpy as np

# Gensim imports
from gensim.models import Doc2Vec

# Scikit-learn imports
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve

# Visualization analysis imports
import matplotlib.pyplot as plt

# Miscellaneous imports
from os import mkdir, getcwd
from os.path import isfile, isdir
from random import shuffle



class Vectorize_Reviews(object):
    """
    The Reviews2Vec class trains a doc2vec model on the corpus of reviews and returns a feature matrix of doc2vec
    vectors for training predictive models.
    """
    def __init__(self, Reviews, model_name: str = 'Doc2VecModel', path: str = None, 
                 vec_size: int =200, min_count: int =10, window: int =10, sample: float =1e-5,
                 negative: int =5, workers: int =4, dm:int =1, epochs: int=10,
                 model: Doc2Vec = None):
        # Define data structures as class variables
        self.Reviews = Reviews
        
        # system processes
        if path is None:
            self.path = getcwd() + '/'

        # Define internal Doc2vec parameters
        self.vec_size = vec_size
        self.min_count = min_count
        self.window = window
        self.sample = sample
        self.negative = negative
        self.workers = workers
        self.dm = dm
        self.modelID = model_name
        self.model = model

        # Set number of training epochs
        self.epochs = epochs


    @staticmethod
    def shuffle_lists(X):
        # Shuffle X and Y while maintaining index coordination
        X_shuf = []
        idx_list = np.arange(len(X)).tolist()
        shuffle(idx_list)
        for idx in idx_list:
            X_shuf.append(X[idx])
        return X_shuf


    def review_to_vec(self, model: Doc2Vec = None, review: list = None):
        # Convert review to vector using trained model
        if model is None:
            if self.model is None:
                print('No Model to use')
                return(False)
            else:
                vec = np.array(self.model.infer_vector(review.words)).reshape((1, self.vec_size)) # TODO CHECK if this flows through correctly
        else:
                vec = np.array(model.infer_vector(review.words)).reshape((1, self.vec_size)) # TODO CHECK if this flows through correctly
        return vec


    def corpus_to_vec(self, model, corpus):
        # Get list of doc2vec vectors from corpus
        vecs = [self.review_to_vec(self.model, review) for review in corpus]

        # Convert list to numpy array
        vec_arr = np.concatenate(vecs)

        return vec_arr


    def build_vocabulary(self):
        # Initialize doc2vec distributed memory object
        self.model = Doc2Vec(min_count=self.min_count,
                                      window=self.window,
                                      vector_size=self.vec_size,
                                      sample=self.sample,
                                      negative=self.negative,
                                      workers=self.workers,
                                      dm=self.dm)
        print("Building vocabulary...\n")
        # Build vocabulary over all reviews
        self.model.build_vocab(self.Reviews)
        return(True)

    def train_on_reviews(self, model: Doc2Vec = None, X: list = None):
        """Run through the dataset multiple times, shuffling the data each time to improve accuracy"""
        
        if X is None:
            X_orig = None
            X = self.Reviews
        else:
            X_orig = X.copy()

        for epoch in range(self.epochs):
            print("Training epoch: {0}/{1}".format(epoch+1, self.epochs))
            if model is None:
                self.model.train(X, total_examples=self.model.corpus_count, epochs=self.epochs)
            else:
                model.train(X, total_examples=model.corpus_count)
            # Shuffle data
            print("Shuffling data...")
            X = self.shuffle_lists(X)

        print("Calculating doc2vec vectors...")
        if X_orig is None:
            doc2vec_vecs = self.corpus_to_vec(model, self.Reviews)
        else:
            doc2vec_vecs = self.corpus_to_vec(model, X_orig)
        print("Done training...\n")
        
        if model is None:
            return(doc2vec_vecs)
        else:
            return(doc2vec_vecs, model)
        self.model.save(model_path + '/' + model_name)
        print('Model Saved')
        return(True)

    def train_doc2vec(self, model: Doc2Vec = None, Data: list = None, force = False):
        """Prep function to train model"""
        print('Initilizng training model')
        
        self.build_vocabulary()
        if model is None:
            print("Training doc2vec model on intenral model...")
            train_vecs = self.train_on_reviews(X = Data)
            return(train_vecs)
        else:
            print("Training doc2vec model on external model...")
            test_vecs, model = self.train_on_reviews(model=model, X = Data)
            return(train_vecs, model)        

    def extract_model(self):
        """returns the model that was generated"""
        return(self.model)

    def load_model(self):
        """loads a doc2vec file"""
        model_file = self.path + 'model/' + self.modelID + '.doc2vec'
        if isfile(model_file):
            self.model = Doc2Vec.load(model_file)
            print('Model Loaded')
            return(True)
        else:
            print('Model %s not found at %s' % (self.modelID + '.doc2vec', self.path + 'model/'))
            return(False)
			
    def save_model(self):
        """saves a doc2vec model"""
        model_path = self.path + 'model' 
        model_name = self.modelID + '.doc2vec'
        if self.model is None:
            print('No model to save')
            return(False)
        if ~isdir(model_path):
            mkdir(model_path)
        self.model.save(model_path + '/' + model_name)
        return(True)

VR = Vectorize_Reviews(df['tagged'])
VR.train_doc2vec()

class Classify_Reviews(object):
    """
    The Review_Classification class trains a logistic regression classifier, employing stochastic gradient descent with L2 regularization.
    """
    def __init__(self, train_vecs, Y_train, test_vecs, Y_test):
        self.train_vecs = train_vecs
        self.Y_train = Y_train
        self.test_vecs = test_vecs
        self.Y_test = Y_test


    @staticmethod
    def persist_model(model):
        # Pickle model for persistence
        with open("models/logreg_model.pkl", 'wb') as output_model:
            cPickle.dump(model, output_model)


    @staticmethod
    def load_model():
        # Load pickled random forest model
        with open("models/logreg_model.pkl", 'rb') as input_model:
            model = cPickle.load(input_model)
        return model


    def train_model(self):
        if not isfile("models/logreg_model.pkl"):
            print "Training logistic regression classifier..."
            # Initialize logistic regression classifier employing stochastic gradient descent with L1 regularization
            lr = SGDClassifier(loss="log", penalty="l1")

            # Define a parameter grid to search over
            param_grid = {"alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]}

            # Setup 5-fold stratified cross validation
            cross_validation = StratifiedKFold(n_splits=5)

            # Perform grid search over hyperparameter configurations with 5-fold cross validation for each
            clf = GridSearchCV(lr,
                               param_grid=param_grid,
                               cv=cross_validation,
                               n_jobs=10,
                               verbose=10)
            clf.fit(self.train_vecs, self.Y_train)

            # Extract best estimator and pickle model for persistence
            self.persist_model(clf.best_estimator_)
        else:
            print "Logistic regression classifier is already trained..."


    def validate_model(self):
        print "Validating classifier..."

        # Load classifier
        model = self.load_model()

        # Classify test dataset
        Y_predicted = model.predict(self.test_vecs)

        # Calculate AUC score
        roc_auc = roc_auc_score(self.Y_test, Y_predicted)

        # Print full classification report
        print classification_report(self.Y_test,
                                    Y_predicted,
                                    target_names=["negative", "positive"])
        print "Area under ROC curve: {:0.3f}".format(roc_auc)

        # Compute ROC curve and area under the curve
        probs = model.predict_proba(self.test_vecs)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.Y_test, probs)

        plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.3f)'%(roc_auc))

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.title('Receiver Operating Characteristic (ROC) curve', fontsize=12)
        plt.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')
        plt.grid(True, linestyle = 'dotted')
        plt.savefig("doc2vec_roc.png")
        print "ROC curve created..."
