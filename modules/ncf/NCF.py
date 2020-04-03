#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/28 9:11 下午
# @Author  : guifeng(moguifeng@baice100.com)
# @File    : NCF.py
import numpy as np
import logging
import argparse
import multiprocessing as mp
from time import time
from tensorflow.keras.layers import Dense, Flatten, Input, Embedding, Multiply, Concatenate
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from evaluate import evaluate_model
from DataLoader import DataLoader
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='/Users/wizardholy/project/recsys_learning/datas/info/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='info',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=2,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            if (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


def get_gmf_model(num_users, num_items, emb_dim):
    gmf_user_input = Input(shape=(1,), dtype="int32", name="user_input")
    gmf_item_input = Input(shape=(1,), dtype="int32", name="item_input")
    gmf_emb_user = Embedding(input_dim=num_users, output_dim=emb_dim, name="user_emb", input_length=1)
    gmf_emb_item = Embedding(input_dim=num_items, output_dim=emb_dim, name="item_emb", input_length=1)
    gmf_user_emb = Flatten()(gmf_emb_user(gmf_user_input))
    gmf_item_emb = Flatten()(gmf_emb_item(gmf_item_input))
    gmb_merge_layer = Multiply()([gmf_user_emb, gmf_item_emb])
    pred = Dense(1, activation='sigmoid', kernel_initializer="lecun_uniform", name="prediction")(gmb_merge_layer)
    model = Model(inputs=[gmf_user_input, gmf_item_input],
                  outputs=pred)
    return model


def get_mlp_model(num_users, num_items, layers=[64, 32, 16, 8]):
    mlp_user_input = Input(shape=(1,), dtype="int32", name="user_input")
    mlp_item_input = Input(shape=(1,), dtype="int32", name="item_input")
    mlp_emb_user = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2),
                             name="user_emb", input_length=1)
    mlp_emb_item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2),
                             name="item_emb", input_length=1)
    mlp_user_emb = Flatten()(mlp_emb_user(mlp_user_input))
    mlp_item_emb = Flatten()(mlp_emb_item(mlp_item_input))
    mlp_vector_layer = Concatenate()([mlp_user_emb, mlp_item_emb])
    for idx in range(1, len(layers)):
        mlp_vector_layer = Dense(layers[idx], activation='relu', kernel_initializer="lecun_uniform",
                                 name="mlp_layer_" + str(idx))(mlp_vector_layer)
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(mlp_vector_layer)
    model = Model(inputs=[mlp_user_input, mlp_item_input],
                  outputs=prediction)
    return model


def get_neumf_model(num_users, num_items, layers=[128, 64, 32, 16], reg=["l1"]):
    user_input = Input(shape=(1,), dtype="int32", name="user_input")
    item_input = Input(shape=(1,), dtype="int32", name="item_input")

    gmf_emb_user = Embedding(input_dim=num_users, output_dim=emb_dim,
                             name="gmf_user_emb", input_length=1)
    gmf_emb_item = Embedding(input_dim=num_items, output_dim=emb_dim,
                             name="gmf_item_emb", input_length=1)
    gmf_user_emb = Flatten()(gmf_emb_user(user_input))
    gmf_item_emb = Flatten()(gmf_emb_item(item_input))
    gmb_vector_layer = Multiply()([gmf_user_emb, gmf_item_emb])

    mlp_emb_user = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2),
                             name="mlp_user_emb", input_length=1)
    mlp_emb_item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2),
                             name="mlp_item_emb", input_length=1)
    mlp_user_emb = Flatten()(mlp_emb_user(user_input))
    mlp_item_emb = Flatten()(mlp_emb_item(item_input))
    mlp_vector_layer = Concatenate()([mlp_user_emb, mlp_item_emb])

    for idx in range(1, len(layers)):
        mlp_vector_layer = Dense(layers[idx], activation='relu', kernel_initializer="lecun_uniform",
                                 name="mlp_layer_" + str(idx))(mlp_vector_layer)

    predict_vector = Concatenate()([gmb_vector_layer, mlp_vector_layer])
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(predict_vector)
    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)
    return model


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logging.warning("start train process")
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    emb_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain

    topK = 10
    evaluation_threads = 1

    t1 = time()
    dataset = DataLoader(args.path + "/" + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))
    print("NeuMF arguments: %s " % (args))
    # model = get_mlp_model(num_users, num_items)
    # model = get_gmf_model(num_users, num_items, emb_dim)
    emb_path = "emb.txt"
    model = get_neumf_model(num_users, num_items)
    type = "neu_mf"
    model_out_file = 'Pretrain/%s_%s_%d_%s_%d.h5' % (args.dataset, type, emb_dim, args.layers, time())
    model.compile(optimizer="adam", loss="binary_crossentropy")
    total_start = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time() - t1))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(num_epochs):
        t1 = time()
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                embs = model.get_layer("mlp_user_emb").embeddings.numpy()
                with open(emb_path, encoding="utf8", mode="w") as f:
                    for i in range(len(embs)):
                        f.write(str(i) + "\t" + ",".join([str(item) for item in embs[i].tolist()]) + "\n")
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)
    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best GMF model is saved to %s" % (model_out_file))
