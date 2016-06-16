import numpy as np
import glob
import codecs
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import preprocessing


np.random.seed(42)  # Seed the pseudo-random generator so the results are reproducible


# -------------------------------------------------
# ---- Turn files into a nicely usable dataset ----
# -------------------------------------------------

def stupid_features(text):
    """Return a very naive feature set: just the length of the text."""
    return [len(text)]


def simple_features(text):
    """Extract some simple numeric features from one email."""
    length = len(text)
    proportion_upper_case = sum(1 for c in text if c.isupper()) / length
    proportion_lower_case = sum(1 for c in text if c.islower()) / length
    proportion_alphanum = sum(1 for c in text if c.isalnum()) / length
    return [length, proportion_upper_case, proportion_lower_case, proportion_alphanum]


def simple_features_extended(text):
    """Extract some more simple numeric features from one email."""
    count_free = text.count("free")
    count_credit = text.count("money")
    count_penis = text.count("penis")
    count_pill = text.count("pill")
    return [count_free, count_credit, count_penis, count_pill] + simple_features(text)


def get_dataset(feature_extraction_function, number_of_emails=5):
    """Read all ham and spam emails from files and return features and labels."""

    # Data downloaded from
    # http://www.aueb.gr/users/ion/data/enron-spam/

    ham_pattern  = "data/enron*/ham/*.txt"
    spam_pattern = "data/enron*/spam/*.txt"

    # Initialise
    features = []
    labels   = []

    # Find all files containing ham and spam emails and concatenate them with the corresponding label
    ham_files = glob.glob(ham_pattern)
    ham_files = ham_files[0:min(number_of_emails, len(ham_files))]
    spam_files = glob.glob(spam_pattern)
    spam_files = spam_files[0:min(number_of_emails, len(spam_files))]
    files = list(zip(ham_files, [0] * len(ham_files))) + list(zip(spam_files, [1] * len(spam_files)))

    for filename, label in files:
        with codecs.open(filename, "r", encoding="utf-8", errors="ignore") as f:
            text = "".join(f.readlines())
            email_features = feature_extraction_function(text)
            labels.append(label)

            features.append(email_features)

    # Convert to Numpy objects because that's what scikit-learn expects
    features = np.asarray(features)
    labels   = np.asarray(labels)

    print("Extracted features using function '%s'" % (feature_extraction_function.__name__))
    print("%d emails processed: %d ham, %d spam" % (len(features), len(ham_files), len(spam_files)))

    return features, labels


# ---------------------------------------
# ---- Magical machine learning part ----
# ---------------------------------------

NUM_EMAILS = 1000

features, labels = get_dataset(simple_features, number_of_emails=NUM_EMAILS)

# ---- First try: logistic regression ----
print("\n---- FIRST MODEL: LOGISTIC REGRESSION ----")
model1 = LogisticRegression()
model1.fit(features, labels)
predicted_labels = model1.predict(features)
training_accuracy = sklearn.metrics.accuracy_score(labels, predicted_labels)
print("Accuracy on training set: %.3f" % training_accuracy)

# Also look at precision and recall
training_precision = sklearn.metrics.precision_score(labels, predicted_labels)
training_recall = sklearn.metrics.recall_score(labels, predicted_labels)
print("Precision: %.3f, recall: %.3f" % (training_precision, training_recall))

# F1-score combines precision and recall so we have a single number to look at
training_f1 = sklearn.metrics.f1_score(labels, predicted_labels)
print("F1 score: %.3f" % training_f1)


# How do we know we didn't overfit? For this, we want to use cross-validation.
print("\n---- CROSS-VALIDATION ----")
NUM_CV_FOLDS = 5
model_cv = LogisticRegression()
scores = cross_validation.cross_val_score(model_cv, features, labels, cv=NUM_CV_FOLDS, scoring='f1')
print("Mean F1 score in %d-fold cross-validation: %.3f" % (NUM_CV_FOLDS, np.mean(scores)))


print("\n---- SECOND MODEL: LOGISTIC REGRESSION WITH EXTENDED FEATURES ----")
features_ext, labels = get_dataset(simple_features_extended, number_of_emails=NUM_EMAILS)
model2 = LogisticRegression()
scores = cross_validation.cross_val_score(model2, features_ext, labels, cv=NUM_CV_FOLDS, scoring='f1')
print("Mean F1 score in %d-fold cross-validation: %.3f" % (NUM_CV_FOLDS, np.mean(scores)))


print("\n---- THIRD MODEL: NONLINEAR SVM WITH EXTENDED FEATURES ----")
features_ext, labels = get_dataset(simple_features_extended, number_of_emails=NUM_EMAILS)
model3 = SVC()
scores = cross_validation.cross_val_score(model3, features_ext, labels, cv=NUM_CV_FOLDS, scoring='f1')
print("Mean F1 score in %d-fold cross-validation: %.3f" % (NUM_CV_FOLDS, np.mean(scores)))


print("\n---- FOURTH MODEL: NONLINEAR SVM WITH EXTENDED NORMALISED FEATURES ----")
scaler = preprocessing.StandardScaler()
features_scaled = scaler.fit_transform(features_ext)
model4 = SVC()  # Let's start with default parameters
scores = cross_validation.cross_val_score(model4, features_scaled, labels, cv=NUM_CV_FOLDS, scoring='f1')
print("Mean F1 score in %d-fold cross-validation: %.3f" % (NUM_CV_FOLDS, np.mean(scores)))


# Now we found a model that works reasonably well: a nonlinear SVM.
# Let's do grid search to find out the best parameters for it.
print("\n---- GRID SEARCH FOR BEST PARAMETERS ----")
parameter_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
model5 = SVC()
gs = grid_search.GridSearchCV(model5, parameter_grid, cv=NUM_CV_FOLDS, verbose=1, scoring='f1')
gs.fit(features_scaled, labels)

# Let's print the result of our grid search
print("Grid search scores for different values of C:")
for score in gs.grid_scores_:
    print(score)


# Let's take the best estimator and make some predictions
print("\n---- USING THE BEST MODEL FOR PREDICTION ----")
best_estimator = gs.best_estimator_

ham_example = """
Subject: christmas break
fyi
- - - - - - - - - - - - - - - - - - - - - - forwarded by shirley crenshaw / hou / ect on 12 / 14 / 99
07 : 51 am - - - - - - - - - - - - - - - - - - - - - - - - - - -
" van t . ngo " on 12 / 04 / 99 11 : 17 : 01 am
to : vince j kaminski / hou / ect @ ect
cc : shirley crenshaw / hou / ect @ ect
subject : christmas break
dear vince ,
as the holidays approach , i am excited by my coming break from classes
but also about the opportunity to see everyone at enron again and to
work with you and them soon . i am writing to let you know that i would
be very happy to work at enron over my break and i would like to plan
out a schedule .
my semester officially ends dec . 20 th but i may be out of town the week
before christmas . i will be available the following three weeks , from
monday , dec . 27 to friday , jan . 14 . please let me know if during those
three weeks , you would like me to work and for what dates you would need
the most help so that we can arrange a schedule that would be most
helpful to you and so that i can contact andrea at prostaff soon .
please let me know if you have any concerns or questions about a
possible work schedule for me .
give my regards to everyone at the office and wishes for a very happy
holiday season ! i look forward to seeing you soon .
sincerely ,
van ngo
ph : 713 - 630 - 8038
- attl . htm
"""

spam_example = """Subject: [ ilug ] bank error in your favor
substantial monthly income makers voucher
income transfer systems / distribution center
pending income amount : up to $ 21 , 000 . 00
good news ! you have made the substancial income makers list . this means you get the entire system and get the opportunity to make up to $ 21 , 000 . 00 a month .
to receive this system , follow this link !
get ready , you will immediately receive all the information needed to make a substantial monthly income .
what are you waiting for ! ! http : / / www . hotresponders . com / cgi - bin / varpro / vartrack . cgi ? t = wendy 7172 : 1
you are receiving this email due to having requested info on internet businesses . if you are not longer looking for one , please click the remove link below .
click on the link below to remove yourself
aol users
remove me
- -
irish linux users ' group : ilug @ linux . ie
http : / / www . linux . ie / mailman / listinfo / ilug for ( un ) subscription information .
list maintainer : listmaster @ linux . ie
"""

ham_example_features = np.asarray(simple_features_extended(ham_example))
# Reshape because scikit-learn complains if you use a 1D shaped vector instead of 2D shaped one
ham_example_features = np.reshape(ham_example_features, (1, -1))
decision1 = best_estimator.predict(scaler.transform(ham_example_features))
print("Do we predict our ham example to be spam? %d" % decision1)

spam_example_features = np.asarray(simple_features_extended(spam_example))
spam_example_features = np.reshape(spam_example_features, (1, -1))
decision2 = best_estimator.predict(scaler.transform(spam_example_features))
print("Do we predict our spam example to be spam? %d" % decision2)