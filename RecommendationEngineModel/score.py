import os
import json
from collections import defaultdict
from pathlib import Path

from azureml.studio.core.io.model_directory import ModelDirectory
from azureml.studio.modules.recommendation.score_svd_recommender.score_svd_recommender import \
    ScoreSVDRecommenderModule, RecommenderPredictionKind
from azureml.studio.common.datatable.data_table import DataTable
from azureml.designer.serving.dagengine.utils import decode_nan
from azureml.designer.serving.dagengine.converter import create_dfd_from_dict


model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'trained_model_outputs')
schema_file_path = Path(model_path) / '_schema.json'
with open(schema_file_path) as fp:
    schema_data = json.load(fp)


def init():
    global model
    model = ModelDirectory.load(load_from_dir=model_path).model


def run(data):
    data = json.loads(data)
    input_entry = defaultdict(list)
    for row in data:
        for key, val in row.items():
            input_entry[key].append(decode_nan(val))

    data_frame_directory = create_dfd_from_dict(input_entry, schema_data)
    score_params = dict(
        learner=model,
        test_data=DataTable.from_dfd(data_frame_directory),
        training_data=None,
        prediction_kind=RecommenderPredictionKind.RatingPrediction)
    result_dfd, = ScoreSVDRecommenderModule().run(**score_params)
    result_df = result_dfd.data_frame
    return json.dumps(result_df.to_dict("list"))
