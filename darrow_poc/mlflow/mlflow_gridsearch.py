import pandas as pd
import sklearn
import warnings
import os
import tempfile
from datetime import datetime
from azureml.core import Experiment, Workspace
import joblib


class LogRandomSearch():
    """ Class / module to log runs from random search

    """
    def __init__(self, ws):
        self.ws = ws

    def log_run(self, gridsearch, experiment_name: str, model_name: str, tags: dict, run, i):
        cv_results = gridsearch.cv_results_

        run.log("folds", gridsearch.cv)

        print("Logging parameters")
        params = list(gridsearch.param_distributions.keys())
        for param in params:
            run.log(param, cv_results["param_%s" % param][i])

        print("Logging metrics")
        for score_name in [score for score in cv_results if 'test_' in score]:
            if 'std' not in score_name:
                run.log(score_name, cv_results[score_name][i])

        print("Logging model")
        # Save the model to the outputs directory for capture
        model_file_name = f'outputs/{model_name}.pkl'
        joblib.dump(value=gridsearch.best_estimator_, filename=model_file_name)
        # upload the model file explicitly into artifacts
        run.upload_file(name=model_file_name, path_or_stream=model_file_name)

        print("Logging CV results matrix")

        tempdir = tempfile.TemporaryDirectory().name
        os.mkdir(tempdir)
        timestamp = datetime.now().isoformat().split(".")[0].replace(":", ".")
        filename = "%s-%s-cv_results.csv" % (model_name, timestamp)
        csv = os.path.join(tempdir, filename)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.DataFrame(cv_results).to_csv(csv, index=False)
        run.upload_file(name='cv_results', path_or_stream=csv)

        print("Logging extra data related to the experiment")
        for tag in tags.keys():
            run.tag(tag, tags[tag])

    def log_results(self, gridsearch, experiment_name: str, model_name: str, tags={}, log_childs=False):
        """Logging of cross validation results to azureml tracking server
        Args:
            experiment_name (str): experiment name
            model_name (str): Name of the model
            tags (dict): Dictionary of extra data and tags (usually features)
        """
        best_index = gridsearch.best_index_

        experiment = Experiment(workspace=self.ws, name=experiment_name)
        parent_run = experiment.start_logging()

        self.log_run(gridsearch, experiment_name, model_name, tags, parent_run, best_index)

        if log_childs:
            for i in range(len(gridsearch.cv_results_['params'])):
                with parent_run.child_run() as run:
                    self.log_run(gridsearch, experiment_name, model_name, tags, run, i)
                    run.complete()

        parent_run.complete()
