from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tensorflow as tf
import tokenization
import os
import modeling
import optimization
from create_pretraining_data import  TrainingInstance
from create_pretraining_data import  create_float_feature
from create_pretraining_data import  create_int_feature



def create_training_instances(tokens, segment_ids, is_random_next,
                              masked_lm_positions, masked_lm_labels):
  """创建一条实例"""
  instances = []
  instance = TrainingInstance(
      tokens=tokens,
      segment_ids=segment_ids,
      is_random_next=is_random_next,
      masked_lm_positions=masked_lm_positions,
      masked_lm_labels=masked_lm_labels)
  instances.append(instance)
  return instances


def write_instance_to_example_files(instances, tokenizer, output_files, max_seq_length,
                                    max_predictions_per_seq):
  """写实例"""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))
  writer_index = 0
  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()
  tf.logging.info("Wrote %d total instances", total_written)



def main(tokens, segment_ids, is_random_next,
        masked_lm_positions, masked_lm_labels,output_files):
    # 字典
    tokenizer = tokenization.FullTokenizer(
        vocab_file="chinese_L-12_H-768_A-12/vocab.txt", do_lower_case=False)
    # 创建实例
    instances = create_training_instances(
        tokens, segment_ids, is_random_next,
        masked_lm_positions, masked_lm_labels)
    # 写入实例
    write_instance_to_example_files(instances, tokenizer, output_files, max_seq_length=27,
                                    max_predictions_per_seq=1)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    tokens=['[CLS]' ,'寄', '[MASK]', '博','通' ,'者' ,'[SEP]' ,'知', '予', '物' ,'外' ,'志' ,'[SEP]']
    segment_ids = [0,0,0,0,0,0,0,1,1,1,1,1,1]
    is_random_next = False
    masked_lm_positions = [2]
    masked_lm_labels = ['居']
    output_files = ['tmp/1.tfrecord']

    main(tokens, segment_ids, is_random_next,
        masked_lm_positions, masked_lm_labels,output_files)