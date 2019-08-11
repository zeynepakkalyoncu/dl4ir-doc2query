import time
import numpy as np
import tensorflow as tf

import metrics
import modeling
import optimization

# Prepare and import BERT modules

import sys
import os

# !test -d bert_repo || git clone https://github.com/nyu-dl/dl4marco-bert dl4marco-bert
# if not 'dl4marco-berto' in sys.path:
#   sys.path += ['dl4marco-bert']

# Prepare for training:
# Specify training data.
# Specify BERT pretrained model
# Specify GS bucket, create output directory for model checkpoints and eval results.

INIT_CHECKPOINT = 'gs://trec_dl_passage_ranking/trained_model/model.ckpt-100000' #@param {type:"string"}
print('***** BERT Init Checkpoint: {} *****'.format(INIT_CHECKPOINT))

OUTPUT_DIR = 'gs://trec_dl_passage_ranking/output' #@param {type:"string"}
assert OUTPUT_DIR, 'Must specify an existing GCS bucket name'
tf.gfile.MakeDirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

# Now we need to specify the input data dir. Should contain the .tfrecord files
# and the supporting query-docids mapping files.
DATA_DIR = 'gs://trec_dl_passage_ranking/' + sys.argv[1] #@param {type:"string"}
print('***** Data directory: {} *****'.format(DATA_DIR))

# Train / evaluate

# Parameters
USE_TPU = True
DO_TRAIN = False  # Whether to run training.
DO_EVAL = True  # Whether to run evaluation.
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 1e-6
NUM_TRAIN_STEPS = 400000
NUM_WARMUP_STEPS = 40000
MAX_SEQ_LENGTH = 512
SAVE_CHECKPOINTS_STEPS = 1000
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8
BERT_CONFIG_FILE = os.path.join('gs://cloud-tpu-checkpoints/bert/uncased_L-24_H-1024_A-16/bert_config.json')
MAX_EVAL_EXAMPLES = None  # Maximum number of examples to be evaluated.
NUM_EVAL_DOCS = 1000  # Number of docs per query in the dev and eval files.
METRICS_MAP = ['MAP', 'RPrec', 'NDCG', 'MRR', 'MRR@10']
FAKE_DOC_ID = '5500000'


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, log_probs)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode,
                 params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(
                "  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, log_probs) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        scaffold_fn = None
        initialized_variable_names = []
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                             init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint,
                                                  assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps,
                use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    "log_probs": log_probs,
                    "label_ids": label_ids,
                },
                scaffold_fn=scaffold_fn)

        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(dataset_path, seq_length, is_training,
                     max_eval_examples=None, num_skip=0):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""

        batch_size = params["batch_size"]
        output_buffer_size = batch_size * 1000

        def extract_fn(data_record):
            features = {
                "query_ids": tf.FixedLenSequenceFeature(
                    [], tf.int64, allow_missing=True),
                "doc_ids": tf.FixedLenSequenceFeature(
                    [], tf.int64, allow_missing=True),
                "label": tf.FixedLenFeature([], tf.int64),
                "len_gt_titles": tf.FixedLenFeature([], tf.int64),
            }
            sample = tf.parse_single_example(data_record, features)

            query_ids = tf.cast(sample["query_ids"], tf.int32)
            doc_ids = tf.cast(sample["doc_ids"], tf.int32)
            label_ids = tf.cast(sample["label"], tf.int32)
            len_gt_titles = tf.cast(sample["len_gt_titles"], tf.int32)

            input_ids = tf.concat((query_ids, doc_ids), 0)

            query_segment_id = tf.zeros_like(query_ids)
            doc_segment_id = tf.ones_like(doc_ids)
            segment_ids = tf.concat((query_segment_id, doc_segment_id), 0)

            input_mask = tf.ones_like(input_ids)

            features = {
                "input_ids": input_ids,
                "segment_ids": segment_ids,
                "input_mask": input_mask,
                "label_ids": label_ids,
                "len_gt_titles": len_gt_titles,
            }
            return features

        dataset = tf.data.TFRecordDataset([dataset_path])
        dataset = dataset.map(
            extract_fn, num_parallel_calls=4).prefetch(output_buffer_size)

        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=1000)
        else:
            if num_skip > 0:
                dataset = dataset.skip(num_skip)

            if max_eval_examples:
                # Use at most this number of examples (debugging only).
                dataset = dataset.take(max_eval_examples)
                # pass

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes={
                "input_ids": [seq_length],
                "segment_ids": [seq_length],
                "input_mask": [seq_length],
                "label_ids": [],
                "len_gt_titles": [],
            },
            padding_values={
                "input_ids": 0,
                "segment_ids": 0,
                "input_mask": 0,
                "label_ids": 0,
                "len_gt_titles": 0,
            },
            drop_remainder=True)

        return dataset

    return input_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not DO_TRAIN and not DO_EVAL:
        raise ValueError(
            "At least one of `DO_TRAIN` or `DO_EVAL` must be True.")

    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)

    if MAX_SEQ_LENGTH > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (MAX_SEQ_LENGTH, bert_config.max_position_embeddings))

    tpu_cluster_resolver = None
    if USE_TPU:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            TPU_ADDRESS)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=OUTPUT_DIR,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=ITERATIONS_PER_LOOP,
            num_shards=NUM_TPU_CORES,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=2,
        init_checkpoint=INIT_CHECKPOINT,
        learning_rate=LEARNING_RATE,
        num_train_steps=NUM_TRAIN_STEPS,
        num_warmup_steps=NUM_WARMUP_STEPS,
        use_tpu=USE_TPU,
        use_one_hot_embeddings=USE_TPU)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=USE_TPU,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        predict_batch_size=EVAL_BATCH_SIZE)

    if DO_TRAIN:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", TRAIN_BATCH_SIZE)
        tf.logging.info("  Num steps = %d", NUM_TRAIN_STEPS)
        train_input_fn = input_fn_builder(
            dataset_path=DATA_DIR + "/dataset_train.tf",
            seq_length=MAX_SEQ_LENGTH,
            is_training=True)
        estimator.train(input_fn=train_input_fn,
                        max_steps=NUM_TRAIN_STEPS)
        tf.logging.info("Done Training!")

    if DO_EVAL:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", EVAL_BATCH_SIZE)

        predictions_path = OUTPUT_DIR + "/msmarco_predictions_dev.tsv"
        total_count = 0
        if tf.gfile.Exists(predictions_path):
            with tf.gfile.Open(predictions_path, "r") as predictions_file:
                total_count = sum(1 for line in predictions_file)
            tf.logging.info(
                "{} examples already processed. Skipping them.".format(
                    total_count))

        query_docids_map = []
        with tf.gfile.Open(
                DATA_DIR + "/query_doc_ids.txt") as ref_file:
            for line in ref_file:
                query_docids_map.append(line.strip().split("\t"))

        max_eval_examples = None
        if MAX_EVAL_EXAMPLES:
            max_eval_examples = MAX_EVAL_EXAMPLES * NUM_EVAL_DOCS

        eval_input_fn = input_fn_builder(
            dataset_path=DATA_DIR + "/dataset.tf",
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            max_eval_examples=max_eval_examples,
            num_skip=total_count)

        # ***IMPORTANT NOTE***
        # The logging output produced by the feed queues during evaluation is very
        # large (~14M lines for the dev set), which causes the tab to crash if you
        # don't have enough memory on your local machine. We suppress this
        # frequent logging by setting the verbosity to WARN during the evaluation
        # phase.
        tf.logging.set_verbosity(tf.logging.WARN)

        result = estimator.predict(input_fn=eval_input_fn,
                                   yield_single_examples=True)
        start_time = time.time()
        results = []
        all_metrics = np.zeros(len(METRICS_MAP))
        example_idx = 0

        for item in result:
            results.append((item["log_probs"], item["label_ids"]))
            total_count += 1

            if len(results) == NUM_EVAL_DOCS:

                log_probs, labels = zip(*results)
                log_probs = np.stack(log_probs).reshape(-1, 2)
                labels = np.stack(labels)

                scores = log_probs[:, 1]
                pred_docs = scores.argsort()[::-1]
                # pred_docs = np.arange(len(pred_docs))

                gt = set(list(np.where(labels > 0)[0]))

                all_metrics += metrics.metrics(
                    gt=gt, pred=pred_docs, metrics_map=METRICS_MAP)

                start_idx = total_count - NUM_EVAL_DOCS
                end_idx = total_count
                query_ids, doc_ids = zip(*query_docids_map[start_idx:end_idx])
                assert len(
                    set(query_ids)) == 1, "Query ids must be all the same."
                query_id = query_ids[0]

                # Workaround to make mode=a work when the file was not yet created.
                mode = "w"
                if tf.gfile.Exists(predictions_path):
                    mode = "a"
                with tf.gfile.Open(predictions_path, mode) as predictions_file:
                    for rank, doc_idx in enumerate(pred_docs):
                        doc_id = doc_ids[doc_idx]
                        predictions_file.write(
                            "\t".join((query_id, doc_id, str(rank + 1))) + "\n")
                example_idx += 1
                results = []

            if total_count % 10000 == 0:
                tf.logging.warn(
                    "Read {} examples in {} secs. Metrics so far:".format(
                        total_count, int(time.time() - start_time)))
                tf.logging.warn("  ".join(METRICS_MAP))
                tf.logging.warn(all_metrics / example_idx)

        # Once the feed queues are finished, we can set the verbosity back to
        # INFO.
        tf.logging.set_verbosity(tf.logging.INFO)

        all_metrics /= example_idx

        tf.logging.info("Final Metrics:")
        tf.logging.info("  ".join(METRICS_MAP))
        tf.logging.info(all_metrics)
        tf.logging.info("Done Evaluating!")


if __name__ == "__main__":
    tf.app.run()
