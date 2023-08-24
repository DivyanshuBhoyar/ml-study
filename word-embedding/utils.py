import os
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector


# projection_data : List[label, vector]
def create_projection(projection_data, path="./"):
    meta_file = "metadata.tsv"
    samples = len(projection_data)
    vector_dim = len(projection_data[0][1])
    projection_matrix = np.zeros((samples, vector_dim))

    # write meta file with labels, create projection_matrix
    with open(os.path.join(path, meta_file), "w") as f:
        for i, row in enumerate(projection_data):
            label, vector = row[0], row[1]
            projection_matrix[i] = vector
            f.write(f"{label}\n")

    weights = tf.Variable(projection_matrix, trainable=False, name="word_embeddings")

    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(path, "embedding.ckpt"))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = meta_file
    projector.visualize_embeddings(path, config)

    tensorboard_command = "tensorboard --logdir=" + path
    subprocess.Popen(tensorboard_command, shell=True)


import tensorflow as tf
import numpy as np
import gensim
import subprocess
import os


def save_word_embeddings3(embedding_vectors, word_vecs):
    tf_w_embeddings = tf.Variable(embedding_vectors, name="word_embeddings")
    checkpt_dir = "./checkpoints"
    log_dir = "./logs"
    # Create the checkpoints directory if it doesn't exist
    if not os.path.exists(checkpt_dir):
        os.makedirs(checkpt_dir)

    checkpoint_prefix = checkpt_dir, "model.ckpt"
    checkpoint = tf.train.Checkpoint(word_embeddings=tf_w_embeddings)
    checkpoint.save(file_prefix=checkpoint_prefix)

    metadata_path = checkpt_dir, "metadata.tsv"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for word, index in word_vecs.key_to_index.items():
            f.write(word + "\n")

    summary_writer = tf.summary.create_file_writer(log_dir)

    with summary_writer.as_default():
        # Reshape the embeddings to 2D
        reshaped_embeddings = tf.reshape(
            tf_w_embeddings, [-1, embedding_vectors.shape[1]]
        )

        # Embed the words with labels
        tf.summary.text(
            "labels", tf.convert_to_tensor(list(word_vecs.key_to_index)), step=0
        )
        tf.summary.write("embeddings", reshaped_embeddings, step=0)

    tensorboard_command = "tensorboard --logdir=" + log_dir
    subprocess.Popen(tensorboard_command, shell=True)
