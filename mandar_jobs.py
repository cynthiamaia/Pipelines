import glob
import os
from sklearn.preprocessing import (StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler,Normalizer,PowerTransformer,QuantileTransformer)
from sklearn.decomposition import KernelPCA
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
dataset_paths = sorted(glob.glob('datasets/*.csv'))
job_number = 1
classifiers = [
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
]

for file_path in dataset_paths:
    for classificador in classifiers:
        job_name = f'job{job_number}'
        print(f'Sending {job_name} with parameters: Dataset = {file_path}')
        python_path = '/Users/moreira/Desktop/cluster_/env/bin' 
        time = '7-00:00:00'
        command = f'sbatch --account=def-menelau --time={time} --job-name={job_name} --cpus-per-task=8 --mem=2G --output="{job_name}_output.txt" --error="{job_name}_error.txt" --wrap="{python_path} V1_pipelines.py {file_path} {classificador}"'
        #print(command)
        os.system(command)
        os.system('sleep 1') # pause to be kind to the scheduler
        job_number +=1

