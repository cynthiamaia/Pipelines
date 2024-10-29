import glob
import os
dataset_paths = sorted(glob.glob('datasets/*.csv'))
job_number = 1
classifiers = [
    'RandomForestClassifier',
    'AdaBoostClassifier',
    'BernoulliNB',
    'DecisionTreeClassifier',
    'ExtraTreesClassifier',
    'GaussianNB',
    'HistGradientBoostingClassifier',
    'KNeighborsClassifier',
    'LinearDiscriminantAnalysis',
    'LinearSVC',
    'SVC',
    'MLPClassifier',
    'MultinomialNB',
    'PassiveAggressiveClassifier',
    'QuadraticDiscriminantAnalysis',
    'SGDClassifier'
]

for file_path in dataset_paths:
    for classificador in classifiers:
        job_name = f'job{job_number}'
        print(f'Sending {job_name} with parameters: Dataset = {file_path}')
        python_path = '/home/cyntmm3/scratch/cluster_/env/bin/python' 
        time = '7-00:00:00'
        command = f'sbatch --account=def-menelau --time={time} --job-name={job_name} --cpus-per-task=8 --mem=2G --output="{job_name}_output.txt" --error="{job_name}_error.txt" --wrap="{python_path} V1_pipelines.py {file_path} {classificador}"'
        #print(command)
        os.system(command)
        os.system('sleep 1') # pause to be kind to the scheduler
        job_number +=1

