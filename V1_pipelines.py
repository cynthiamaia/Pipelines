import sys
argv = sys.argv

file_path = argv[1]
classificador = argv[2]
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import (StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler,Normalizer,PowerTransformer,QuantileTransformer)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.feature_selection import SelectPercentile, GenericUnivariateSelect
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import (BernoulliNB, GaussianNB, MultinomialNB)
from sklearn.linear_model import (LogisticRegression, SGDClassifier, PassiveAggressiveClassifier)
from sklearn.svm import SVC
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import pandas as pd
import logging
import os
import numpy as np
import time

import logging
from datetime import datetime

log_directory = 'logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
    
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(log_directory, f"log_{current_time}.log")
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def preprocess(X_train, X_test, y_train, imputer_strategy, categorical_strategy, scaler, feature_selection):
    # Imputação
    logger.info("Iniciando o pré-processamento dos dados.")
    if imputer_strategy == "simpleimputer":
        numeric_cols = [col for col in X_train.columns if X_train[col].dtype in [np.int64, np.float64]]
        cols_with_nan = [col for col in numeric_cols if X_train[col].isna().any()]

        categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
        cols_with_nan_categorical = [col for col in categorical_cols if X_train[col].isna().any()]
        logger.info(f"Colunas numéricas com NaN: {cols_with_nan}")
        logger.info(f"Colunas categóricas com NaN: {cols_with_nan_categorical}")
        
        if cols_with_nan:
            imputer = SimpleImputer()
            imputer.fit(X_train[cols_with_nan])
            X_train[cols_with_nan] = imputer.transform(X_train[cols_with_nan])
            X_test[cols_with_nan] = imputer.transform(X_test[cols_with_nan])  # Imputação no conjunto de teste
            logger.info("Imputação de NaNs nas colunas numéricas concluída.")

        if categorical_cols:
            imputer_categorical = SimpleImputer(strategy='constant', fill_value='missing')
            imputer_categorical.fit(X_train[categorical_cols])
            X_train[categorical_cols] = imputer_categorical.transform(X_train[categorical_cols])
            X_test[categorical_cols] = imputer_categorical.transform(X_test[categorical_cols])
            logger.info(f"Imputação de NaNs nas colunas categóricas concluída: {X_train.isna().any()}")
    
    # Codificação
    if categorical_strategy == "onehot":
        categorical_cols = [i for i in range(X_train.shape[1]) if X_train.iloc[:, i].dtype == 'object']
        if categorical_cols:
            encoder = OneHotEncoder()
            X_train_categorical = encoder.fit_transform(X_train.iloc[:, categorical_cols])
            X_train_dense = X_train_categorical.toarray()
            X_test_categorical = encoder.transform(X_test.iloc[:, categorical_cols])           
            # Concatenar os dados categóricos codificados de volta
            X_test_dense = X_test_categorical.toarray()
            X_train = np.hstack((X_train_dense, X_train.drop(X_train.columns[categorical_cols], axis=1).to_numpy()))
            X_test = np.hstack((X_test_dense, X_test.drop(X_test.columns[categorical_cols], axis=1).to_numpy()))
            logger.info("Codificação OneHot concluída.")

    elif categorical_strategy == "ordinalencoder":
        for i in range(X_train.shape[1]):
            if X_train.iloc[:, i].dtype == 'object':
                encoder = OrdinalEncoder()
                X_train.iloc[:, i] = encoder.fit_transform(X_train.iloc[:, [i]])
                X_test.iloc[:, i] = encoder.transform(X_test.iloc[:, [i]])  # Aplicar no conjunto de teste
                logger.info("Codificação Ordinal concluída.")
    # Aplicar escaladores
    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "minmax":
        scaler_obj = MinMaxScaler()
    elif scaler == "robust":
        scaler_obj = RobustScaler()
    elif scaler == "normalizer":
        scaler_obj = Normalizer()
    elif scaler == "power":
        scaler_obj = PowerTransformer()
    elif scaler == "quantile":
        scaler_obj = QuantileTransformer()
    else:
        scaler_obj = None

    if scaler_obj is not None:
        X_train = scaler_obj.fit_transform(X_train)
        X_test = scaler_obj.transform(X_test)  # Aplicar o mesmo scaler ao conjunto de teste
        logger.info(f"Escalamento usando {scaler} concluído.")

    # PCA
    if feature_selection == "pca":
        pca = PCA() 
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        logger.info(f"Transformação PCA aplicada o fit_transform do PCA: {X_train}")
        logger.info(f"Transformação PCA aplicada o transform do PCA: {X_test}")
        logger.info("Transformação PCA aplicada.")
        
    elif feature_selection == "fastica":
        ica = FastICA()
        X_train = ica.fit_transform(X_train)
        X_test = ica.transform(X_test)
        logger.info(f"Transformação FastICA aplicada o fit_transform do FastICA: {X_train}")
        logger.info(f"Transformação FastICA aplicada o transform do FastICA: {X_test}")
        logger.info("Transformação FastICA aplicada.")
        
    elif feature_selection == "truncated_svd":
        svd = TruncatedSVD()
        X_train = svd.fit_transform(X_train)
        X_test = svd.transform(X_test)
        logger.info(f"Transformação TruncatedSVD aplicada o fit_transform do TruncatedSVD: {X_train}")
        logger.info(f"Transformação TruncatedSVD aplicada o transform do TruncatedSVD: {X_test}")
        logger.info("Transformação TruncatedSVD aplicada.")
        
    elif feature_selection == "select_percentile":
        selector = SelectPercentile()
        X_train = selector.fit_transform(X_train, y_train) 
        X_test = selector.transform(X_test)
        logger.info(f"Transformação SelectPercentile aplicada o fit_transform do SelectPercentile: {X_train}")
        logger.info(f"Transformação SelectPercentile aplicada o transform do SelectPercentile: {X_test}")
        logger.info("Transformação SelectPercentile aplicada.")

    elif feature_selection == "generic_univariate":
        selector = GenericUnivariateSelect() 
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        logger.info(f"Transformação GenericUnivariateSelect aplicada o fit_transform do GenericUnivariateSelect: {X_train}")
        logger.info(f"Transformação GenericUnivariateSelect aplicada o transform do GenericUnivariateSelect: {X_test}")
        logger.info("Transformação GenericUnivariateSelect aplicada.")

    elif feature_selection == "polynomial_features":
        poly = PolynomialFeatures()
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)
        logger.info(f"Transformação PolynomialFeatures aplicada o fit_transform do PolynomialFeatures: {X_train}")
        logger.info(f"Transformação PolynomialFeatures aplicada o transform do PolynomialFeatures: {X_test}")
        logger.info("Transformação PolynomialFeatures aplicada.")

    elif feature_selection == "nystroem":
        nystroem = Nystroem() 
        X_train = nystroem.fit_transform(X_train)
        X_test = nystroem.transform(X_test)
        logger.info(f"Transformação Nystroem aplicada o fit_transform do Nystroem: {X_train}")
        logger.info(f"Transformação Nystroem aplicada o transform do Nystroem: {X_test}")
        logger.info("Transformação Nystroem aplicada.")

    elif feature_selection == "rbf_sampler":
        rbf_sampler = RBFSampler()
        X_train = rbf_sampler.fit_transform(X_train)
        X_test = rbf_sampler.transform(X_test)
        logger.info(f"Transformação RBFSampler aplicada o fit_transform do RBFSampler: {X_train}")
        logger.info(f"Transformação RBFSampler aplicada o transform do RBFSampler: {X_test}")
        logger.info("Transformação RBFSampler aplicada.")

    elif feature_selection == "random_trees_embedding":
        rtb = RandomTreesEmbedding()
        X_train = rtb.fit_transform(X_train)
        X_test = rtb.transform(X_test)
        logger.info(f"Transformação RandomTreesEmbedding aplicada o fit_transform do RandomTreesEmbedding: {X_train}")
        logger.info(f"Transformação RandomTreesEmbedding aplicada o transform do RandomTreesEmbedding: {X_test}")
        logger.info("Transformação RandomTreesEmbedding aplicada.")

    elif feature_selection == "feature_agglomeration":
        agglomerator = FeatureAgglomeration()  
        X_train = agglomerator.fit_transform(X_train)
        X_test = agglomerator.transform(X_test)
        logger.info(f"Transformação FeatureAgglomeration aplicada o fit_transform do FeatureAgglomeration: {X_train}")
        logger.info(f"Transformação FeatureAgglomeration aplicada o transform do FeatureAgglomeration: {X_test}")
        logger.info("Transformação FeatureAgglomeration aplicada.")
    
    elif feature_selection == "extra_tree":
        model = ExtraTreesClassifier()
        model.fit(X_train, y_train)
        selector = SelectFromModel(model)  
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        logger.info(f"Transformação ExtraTreesClassifier aplicada o fit_transform do ExtraTreesClassifier: {X_train}")
        logger.info(f"Transformação ExtraTreesClassifier aplicada o transform do ExtraTreesClassifier: {X_test}")
        logger.info("Transformação ExtraTreesClassifier aplicada.")

    elif feature_selection == "linear_svc":
        model = LinearSVC()
        model.fit(X_train, y_train)
        selector = SelectFromModel(model)  
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        logger.info(f"Transformação LinearSVC aplicada o fit_transform do LinearSVC: {X_train}")
        logger.info(f"Transformação LinearSVC aplicada o transform do LinearSVC: {X_test}")
        logger.info("Transformação LinearSVC aplicada.")
        
    elif feature_selection == "kernel_pca":
        kernel_pca = KernelPCA()  # Você pode ajustar o kernel conforme necessário
        X_train = kernel_pca.fit_transform(X_train)
        X_test = kernel_pca.transform(X_test)
        logger.info(f"Transformação KernelPCA aplicada o fit_transform do KernelPCA: {X_train}")
        logger.info(f"Transformação KernelPCA aplicada o transform do KernelPCA: {X_test}")
        logger.info("Transformação KernelPCA aplicada.")

    return X_train, X_test

def process_dataset(file_path):
    ds = pd.read_csv(file_path)
    dataset_name = os.path.basename(file_path).replace('.csv', '')

    X = ds.iloc[:, :-1]
    y = ds.iloc[:, -1].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    cont = 0
    training_time = 0
    testing_time = 0

    # Loop para as combinações de hiperparâmetros
    for combination in combinations:
        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            try:
                cont += 1
                imputer, categorical_strategy, scaler, feature_selection, classifier = combination
                print(f'Tentando combinação {cont} - Imputer: {imputer}, Categorização: {categorical_strategy}, Seleção de Feature: {feature_selection}, Scaler: {scaler}, Modelo: {classifier}')
                nome_combinacao = f"{imputer}_{categorical_strategy}_{scaler}_{feature_selection}_{classifier}"

                # Dividir em conjuntos de treino e teste
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Pré-processar os dados
                X_train_processed, X_test_processed = preprocess(X_train, X_test, y_train, imputer, categorical_strategy, scaler, feature_selection)
                print(X_train_processed.shape)
                print(X_test_processed.shape)
                logger.info(f"Os dados de treinamento da variavel X_train_processed {X_train_processed}")
                logger.info(f"Os dados de teste da variavel X_test_processed {X_test_processed}")
                
                # Treinamento do modelo
                start_time = time.time()
                classifier.fit(X_train_processed, y_train)
                training_time = time.time() - start_time
                start_time = time.time()
                y_pred = classifier.predict(X_test_processed)
                testing_time = time.time() - start_time
                acc = accuracy_score(y_test, y_pred)
                results.append({'Dataset': dataset_name,
                                'Imputer Strategy': imputer,
                                'Categorical Strategy': categorical_strategy,
                                'Feature Selection': feature_selection,
                                'Scaler': scaler,
                                'Classifier': classifier,
                                'Fold': fold,
                                'Fold Accuracy': acc,
                                'Error': None,
                               'Training Time': training_time,
                               'Testing Time': testing_time})

                logger.info(f'Combinação {cont} - Modelo: {classifier} com acurácia {acc}')
                predictions_df = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
                output_dir = 'datasets_results_1'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                dataset_results_dir = os.path.join(output_dir, dataset_name)
                os.makedirs(dataset_results_dir, exist_ok=True)
                predictions_file = os.path.join(dataset_results_dir, f'predictions_fold_dataset_{dataset_name}_combination_{nome_combinacao}_fold_{fold}.csv')
                predictions_df.to_csv(predictions_file, index=False)
            except Exception as e:
                error_msg = (f"Erro na combinação {cont} - "
                             f"Imputer: {imputer}, Categorização: {categorical_strategy}, Seleção de Feature: {feature_selection}, "
                             f"Scaler: {scaler}, Fold: {fold} - Exception: {e}")
                logger.error(error_msg)
                results.append({'Dataset': dataset_name,
                                'Imputer Strategy': imputer,
                                'Categorical Strategy': categorical_strategy,
                                'Feature Selection': feature_selection,
                                'Scaler': scaler,
                                'Classifier': classifier,
                                'Fold': fold,
                                'Fold Accuracy': None,
                                'Error': error_msg, 
                                'Training Time': training_time,
                                'Testing Time': testing_time})

    # Salvando os resultados gerais
    results_df = pd.DataFrame(results)
    output_dir = 'datasets_results_1'
    os.makedirs(output_dir, exist_ok=True)
    dataset_results_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_results_dir, exist_ok=True)
    output_file = os.path.join(dataset_results_dir, f'{dataset_name}_results_{classificador}.csv')
    results_df.to_csv(output_file, index=False)
    logger.info(f'RESULTADOS SALVOS NO CAMINHO {output_file}')

# Chame a função process_dataset com o caminho do arquivo CSV

# Testar todas as combinações
imputation_strategy = [None, 'simpleimputer']
categorical_strategies = [None, 'onehot','ordinalencoder']
scalers = [None, 'standard', "minmax","robust","normalizer", "power","quantile"]
feature_selections = [None, 'pca']
'''classifiers = [
    RandomForestClassifier(),
    AdaBoostClassifier(),
    BernoulliNB(),
    DecisionTreeClassifier(),
    ExtraTreesClassifier(),
    GaussianNB(),
    HistGradientBoostingClassifier(),
    KNeighborsClassifier(),
    LinearDiscriminantAnalysis(),
    LinearSVC(),
    SVC(),
    MLPClassifier(),
    MultinomialNB(),
    PassiveAggressiveClassifier(),
    QuadraticDiscriminantAnalysis(),
    SGDClassifier()
] '''
combinations = []
for imputer in imputation_strategy:
    for categorical in categorical_strategies:
        for scaler in scalers:
            for feature in feature_selections:
                #for classificador in classifiers:
                combinations.append((imputer, categorical, scaler, feature, classificador))

process_dataset(file_path)