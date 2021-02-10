import os
import pprint
import tempfile
import math
from typing import Dict

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_hub as hub

import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

from apache_beam.transforms import util

from tensorflow.python.keras.models import Model, load_model

raw_data = [
      {'title': ['sterling silver necklace'], 'query': ['silver necklace']},
      {'title': ['gold bracelet'], 'query': ['silver necklace']},
      {'title': ['opal'], 'query': ['silver necklace']}
  ]

raw_feature_spec = {
    'title': tf.io.VarLenFeature(tf.string),
    'query': tf.io.VarLenFeature(tf.string),
}

raw_data_metadata = dataset_metadata.DatasetMetadata(
  schema_utils.schema_from_feature_spec(raw_feature_spec)
)

def get_embedding_similarity(input_1, input_2):
    hub_url = 'https://tfhub.dev/google/nnlm-en-dim50/2'
    embed = hub.load(hub_url)

    @tf.function
    def embed_fn(a, b):
        text_1_embedding = embed(a)
        text_2_embedding = embed(b)

        text_1_normalized = tf.nn.l2_normalize(text_1_embedding, axis=-1)
        text_2_normalized = tf.nn.l2_normalize(text_2_embedding, axis=-1)
        cosine_distance = tf.reduce_sum(
            tf.multiply(text_1_normalized, text_2_normalized), axis=-1
        )
        clip_cosine_similarities = tf.clip_by_value(cosine_distance, -1.0, 1.0)
        cosine_scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi

        l2_norm = tf.norm(text_1_normalized - text_2_normalized, axis=-1, ord='euclidean')

        return cosine_scores, l2_norm
    return embed_fn(input_1, input_2)

def impute(feature_tensor, default):
    sparse = tf.sparse.SparseTensor(
        feature_tensor.indices,
        feature_tensor.values,
        [feature_tensor.dense_shape[0], 1],
    )
    dense = tf.sparse.to_dense(sp_input=sparse, default_value=default)

    return tf.squeeze(dense, axis=1)

def text_feature_transform(feature_dict: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    outputs = dict()
    for key in feature_dict.keys():
        feature = impute(feature_dict[key], "")
        outputs[key] = feature
    return outputs

def preprocessing_fn(inputs):
    inputs_text_transform = {key: inputs[key] for key in ['query','title']}
    outputs_text = text_feature_transform(inputs_text_transform)

    outputs = {
        **outputs_text,
    }

    cos_dist, euc_dist = get_embedding_similarity(impute(inputs['query'],""), impute(inputs['title'],""))
    outputs[f"query_title_nnlm_cos_dist"] = cos_dist
    outputs[f"query_title_nnlm_euc_dist"] = euc_dist

    return outputs

def run_tft_pipeline():
    temp_dir = tempfile.mkdtemp()
    os.environ['TFHUB_CACHE_DIR']=os.path.join(temp_dir, "tfhub_modules")
    with tft_beam.Context(temp_dir=temp_dir, force_tf_compat_v1=False):
        transformed_dataset, transform_fn = ((raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))

    transformed_data, transformed_metadata = transformed_dataset

    print('\nRaw data:\n{}\n'.format(pprint.pformat(raw_data)))
    print('Transformed data:\n{}'.format(pprint.pformat(transformed_data)))


if __name__== "__main__":
    run_tft_pipeline()


