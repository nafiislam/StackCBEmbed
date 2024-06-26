import pickle;
import skops.io as sio
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier
from sklearn.svm import SVC

def preprocess_the_dataset(feature_X):

    pt = PowerTransformer()
    pt.fit(feature_X)
    feature_X = pt.transform(feature_X)

    return feature_X

def load_the_pickle_files_base_layer(converted_all_features_with_output, choice):

    test_X = preprocess_the_dataset(converted_all_features_with_output)
    test_base_output_total = np.zeros((len(test_X), 1), dtype=float)

    if choice == 1:
        pickle_folder_path = str('../base_layer_pickle_files_with_PSSM/')
    else:
        pickle_folder_path = str('../base_layer_pickle_files_without_PSSM/')

    base_classifiers = ['SVM', 'MLP', 'ET', 'PLS']
    
    for i in range(0, 10):
        for base_classifier in base_classifiers:
            pickle_file_path = str(pickle_folder_path + base_classifier + '_base_layer_' + str(i) + '.sav')
            #print(pickle_file_path)

            outfile = open(pickle_file_path, 'rb')
            model = pickle.load(outfile)
            outfile.close()

            if base_classifier == 'PLS':
                p_all = []
                p_all.append([1 - np.abs(item[0]) for item in model.predict(test_X)])
                p_all.append([np.abs(item[0]) for item in model.predict(test_X)])
                y_proba = np.transpose(np.array(p_all))[:, 1].reshape(-1, 1)
            else:
                y_proba = model.predict_proba(test_X)[:, 1].reshape(-1, 1)
            test_base_output_total = np.concatenate((test_base_output_total, y_proba), axis=1)

    test_base_output_total = np.delete( test_base_output_total, 0, axis=1 )

    return test_base_output_total


def load_the_pickle_files_meta_layer(converted_all_features_with_output,choice):
    test_X = preprocess_the_dataset(converted_all_features_with_output)

    if (choice == 1):
        pickle_file_path = str('../base_layer_pickle_files_with_PSSM/SVM_meta_layer.sav')
    else:
        pickle_file_path = str('../base_layer_pickle_files_without_PSSM/SVM_meta_layer.sav')

    outfile = open(pickle_file_path, 'rb')
    clf = pickle.load(outfile)
    outfile.close()

    y_pred = clf.predict(test_X)

    return y_pred
