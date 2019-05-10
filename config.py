# -*- coding: utf-8 -*-

import os
import sys
import datetime
import numpy as np
from tqdm import tqdm
import sklearn.metrics
import torch
import torch.optim as optim
from torch.autograd import Variable
import ipdb


def to_var(x):
    return Variable(torch.from_numpy(x).cuda())


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


class Config(object):
    def __init__(self, batch_size=64, max_epoch=15, num_class=15, max_length=120, drop_prob=0.5, is_training=True):
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.data_path = './_data'  # path to save data
        self.checkpoint_dir = './_checkpoint'  # path to save models
        self.test_result_dir = './_test_result'    # path to save results

        self.use_bag = True
        self.use_gpu = True
        self.trainModel = None
        self.testModel = None
        self.is_training = is_training
        self.pretrain_model = None  # pre-trained model path

        self.num_classes = num_class
        self.batch_size = batch_size
        self.max_length = max_length
        self.word_size = 200
        self.pos_num = 2 * self.max_length
        self.pos_size = 5
        self.window_size = 3
        self.hidden_size = 200

        self.opt_method = "SGD"
        self.optimizer = None
        self.learning_rate = 0.5
        self.weight_decay = 1e-5
        self.lr_decay = 0.0
        self.drop_prob = drop_prob

        self.max_epoch = max_epoch
        self.save_epoch = 1  # save training model in every save_epoch
        self.valid_epoch = 1  # do validation in every valid_epoch
        self.epoch_range = None

    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model

    def set_epoch_range(self, epoch_range):
        self.epoch_range = epoch_range

    def load_train_data(self):
        print("Reading training data...")
        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
        self.data_train_word = np.load(os.path.join(self.data_path, 'train_word.npy'))
        self.data_train_pos1 = np.load(os.path.join(self.data_path, 'train_pos1.npy'))
        self.data_train_pos2 = np.load(os.path.join(self.data_path, 'train_pos2.npy'))
        self.data_train_mask = np.load(os.path.join(self.data_path, 'train_mask.npy'))
        if self.use_bag:
            self.data_query_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))
            self.data_train_label = np.load(os.path.join(self.data_path, 'train_bag_label.npy'))
            self.data_train_scope = np.load(os.path.join(self.data_path, 'train_bag_scope.npy'))
        else:
            self.data_train_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))
            self.data_train_scope = np.load(os.path.join(self.data_path, 'train_ins_scope.npy'))
        print("Finish reading")
        self.train_order = list(range(len(self.data_train_label)))
        self.train_batches = len(self.data_train_label) / self.batch_size
        if len(self.data_train_label) % self.batch_size != 0:
            self.train_batches += 1

    def load_test_data(self):
        print("Reading testing data...")
        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
        self.data_test_word = np.load(os.path.join(self.data_path, 'test_word.npy'))
        self.data_test_pos1 = np.load(os.path.join(self.data_path, 'test_pos1.npy'))
        self.data_test_pos2 = np.load(os.path.join(self.data_path, 'test_pos2.npy'))
        self.data_test_mask = np.load(os.path.join(self.data_path, 'test_mask.npy'))
        if self.use_bag:
            self.data_test_label = np.load(os.path.join(self.data_path, 'test_bag_label.npy'))
            self.data_test_scope = np.load(os.path.join(self.data_path, 'test_bag_scope.npy'))
        else:
            self.data_test_label = np.load(os.path.join(self.data_path, 'test_ins_label.npy'))
            self.data_test_scope = np.load(os.path.join(self.data_path, 'test_ins_scope.npy'))
        print("Finish reading")
        self.test_batches = len(self.data_test_label) / self.batch_size
        if len(self.data_test_label) % self.batch_size != 0:
            self.test_batches += 1
        self.total_recall = self.data_test_label[:, 1:].sum()

    def load_predict_data(self, d, embeddings):
        self.data_word_vec = embeddings
        self.data_test_word = d.get("word")
        self.data_test_pos1 = d.get("pos1")
        self.data_test_pos2 = d.get("pos2")
        self.data_test_mask = d.get("mask")
        if self.use_bag:
            self.data_test_label = d.get("bag_label")
            self.data_test_scope = d.get("bag_scope")
        else:
            self.data_test_label = d.get("ins_label")
            self.data_test_scope = d.get("ins_scope")
        self.test_batches = len(self.data_test_label) / self.batch_size
        if len(self.data_test_label) % self.batch_size != 0:
            self.test_batches += 1

    def set_train_model(self, model):
        print("Initializing training model...")
        self.model = model
        self.trainModel = self.model(config=self)
        if self.pretrain_model != None:
            self.trainModel.load_state_dict(torch.load(self.pretrain_model))
        self.trainModel.cuda()
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr = self.learning_rate, lr_decay = self.lr_decay, weight_decay = self.weight_decay)
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        else:
            self.optimizer = optim.SGD(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        print("Finish initializing")

    def set_test_model(self, model):
        print("Initializing test model...")
        self.model = model
        self.testModel = self.model(config=self)
        self.testModel.cuda()
        self.testModel.eval()
        print("Finish initializing")

    def set_predict_model(self, model, epoch=None):
        print("Initializing predict model...")
        self.model = model
        self.testModel = self.model(config=self)
        self.testModel.cuda()
        self.testModel.eval()
        path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
        if not os.path.isfile(path):
            raise Exception("[ERROR] pre-trained model doesn't exist.")
        self.testModel.load_state_dict(torch.load(path))
        print("Finish initializing")

    def get_train_batch(self, batch):
        input_scope = np.take(self.data_train_scope,
                              self.train_order[batch * self.batch_size: (batch + 1) * self.batch_size], axis=0)
        index = []
        scope = [0]
        for num in input_scope:
            index = index + list(range(num[0], num[1] + 1))
            scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
        self.batch_word = self.data_train_word[index, :]
        self.batch_pos1 = self.data_train_pos1[index, :]
        self.batch_pos2 = self.data_train_pos2[index, :]
        self.batch_mask = self.data_train_mask[index, :]
        self.batch_label = np.take(self.data_train_label,
                                   self.train_order[batch * self.batch_size: (batch + 1) * self.batch_size], axis=0)
        self.batch_attention_query = self.data_query_label[index]
        self.batch_scope = scope

    def get_test_batch(self, batch):
        input_scope = self.data_test_scope[batch * self.batch_size: (batch + 1) * self.batch_size]
        index = []
        scope = [0]
        for num in input_scope:
            index = index + list(range(num[0], num[1] + 1))
            scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
        self.batch_word = self.data_test_word[index, :]
        self.batch_pos1 = self.data_test_pos1[index, :]
        self.batch_pos2 = self.data_test_pos2[index, :]
        self.batch_mask = self.data_test_mask[index, :]
        self.batch_scope = scope

    def train_one_step(self):
        self.trainModel.embedding.word = to_var(self.batch_word)
        self.trainModel.embedding.pos1 = to_var(self.batch_pos1)
        self.trainModel.embedding.pos2 = to_var(self.batch_pos2)
        self.trainModel.encoder.mask = to_var(self.batch_mask)
        self.trainModel.selector.scope = self.batch_scope
        self.trainModel.selector.attention_query = to_var(self.batch_attention_query)
        self.trainModel.selector.label = to_var(self.batch_label)
        self.trainModel.classifier.label = to_var(self.batch_label)
        self.optimizer.zero_grad()
        loss, _output = self.trainModel()
        loss.backward()
        self.optimizer.step()
        for i, prediction in enumerate(_output):
            if self.batch_label[i] == 0:
                self.acc_NA.add(prediction == self.batch_label[i])
            else:
                self.acc_not_NA.add(prediction == self.batch_label[i])
            self.acc_total.add(prediction == self.batch_label[i])
        return loss.data.item()

    def test_one_step(self):
        self.testModel.embedding.word = to_var(self.batch_word)
        self.testModel.embedding.pos1 = to_var(self.batch_pos1)
        self.testModel.embedding.pos2 = to_var(self.batch_pos2)
        self.testModel.encoder.mask = to_var(self.batch_mask)
        self.testModel.selector.scope = self.batch_scope
        return self.testModel.test()

    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_auc = 0.0
        best_p = None
        best_r = None
        best_epoch = 0
        for epoch in range(self.max_epoch):
            print('Epoch ' + str(epoch) + ' starts...')
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            np.random.shuffle(self.train_order)
            for batch in range(int(self.train_batches)):
                self.get_train_batch(batch)
                loss = self.train_one_step()
                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write("epoch %d step %d time %s | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (epoch, batch, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
                sys.stdout.flush()
            if (epoch + 1) % self.save_epoch == 0:
                print('Epoch ' + str(epoch) + ' has finished')
                print('Saving model...')
                path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
                torch.save(self.trainModel.state_dict(), path)
                print('Have saved model to ' + path)
            if (epoch + 1) % self.valid_epoch == 0:
                self.testModel = self.trainModel
                auc, pr_x, pr_y = self.test_one_epoch()
                if auc > best_auc:
                    best_auc = auc
                    best_p = pr_x
                    best_r = pr_y
                    best_epoch = epoch
        print("Finish training")
        print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
        print("Storing best result...")
        if not os.path.isdir(self.test_result_dir):
            os.mkdir(self.test_result_dir)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
        print("Finish storing")

    def test_one_epoch(self):
        test_score = []
        for batch in tqdm(range(int(self.test_batches))):
            self.get_test_batch(batch)
            batch_score = self.test_one_step()
            test_score = test_score + batch_score
        test_result = []
        for i in range(len(test_score)):
            for j in range(1, len(test_score[i])):
                test_result.append([self.data_test_label[i][j], test_score[i][j]])  # ith sample jth class
        test_result = sorted(test_result, key=lambda x: x[1])
        test_result = test_result[::-1]
        pr_x = []
        pr_y = []
        correct = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / self.total_recall)
        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        print("auc: ", auc)
        return auc, pr_x, pr_y

    def test(self):
        best_epoch = None
        best_auc = 0.0
        best_p = None
        best_r = None
        for epoch in self.epoch_range:
            path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
            if not os.path.exists(path):
                continue
            print("Start testing epoch %d" % epoch)
            self.testModel.load_state_dict(torch.load(path))
            auc, p, r = self.test_one_epoch()
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                best_p = p
                best_r = r
            print("Finish testing epoch %d" % epoch)
        print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
        print("Storing best result...")
        if not os.path.isdir(self.test_result_dir):
            os.mkdir(self.test_result_dir)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
        print("Finish storing")

    def predict(self, id2rel):
        # input: id2rel = {class_name: index}
        # output: lst_pre = [sample_1_pre, sample_2_pre, ...], sample_i_pre = (class_name, score)
        test_score = list()
        for batch in tqdm(range(int(self.test_batches))):
            self.get_test_batch(batch)
            batch_score = self.test_one_step()  # batch_score = [sample_1, sample_2, ...], where sample = np.array(num_class,), i.e., sample is the predicted vector for one-hot
            test_score.extend(batch_score)  # test_score = [sample_1, sample_2, ...], where sample = np.array(num_class,), i.e., sample is the predicted vector for one-hot
        # make prediction
        lst_pre = list()
        for sample in test_score:
            i_pre = int(np.argmax(sample))
            pre = (id2rel.get(i_pre), sample[i_pre])
            lst_pre.append(pre)
        return lst_pre

