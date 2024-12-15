import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from utils import static_utils
import logging
import os
import scipy

from Decomposer.Gram_Schmidt import Gram_Schmidt
from Decomposer.SVD import SVD
from MOO.MGDA import MGDA
from MOO.CAGrad import CAGrad
from MOO.PCGrad import PCGrad
from MOO.DB_MTL import DB_MTL
from MOO.ExcessMTL import ExcessMTL
from MOO.FairGrad import FairGrad
from MOO.MoCo import MoCo
import numpy as np

from SAM_function.TRAM import TRAM
from SAM_function.FSAM import FSAM

import time

class BasicTrainer:
    def __init__(self, model, epoch_threshold = 150, model_name='NeuroMax', use_SAM=1, SAM_name='TRAM', epochs=200, 
                 use_decompose=1, decompose_name='Gram_Schmidt', use_MOO=1, MOO_name='PCGrad', task_num=3,
                 learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125, log_interval=5, learn=0,
                    rho = 0.005, threshold=10, device='cuda', sigma=0.1, lmbda=0.9, acc_step=8):
        self.model = model
        self.epoch_threshold = epoch_threshold
        self.model_name = model_name
        self.task_num = task_num
        self.learn = learn

        self.use_decompose = use_decompose
        self.decompose_name = decompose_name
        self.use_MOO = use_MOO
        self.MOO_name = MOO_name
        self.SAM_name = SAM_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        self.threshold = threshold
        self.use_SAM = use_SAM

        self.rho = rho 
        self.device = device
        self.sigma = sigma
        self.lmbda = lmbda
        self.acc_step = acc_step
        self.logger = logging.getLogger('main')

        self.loss_out = []

        # Thêm ctr_loss
        # self.cluster_distribution = cluster_distribution 
        # self.cluster_mean = cluster_mean 
        # self.topic_embeddings = topic_embeddings
        # self.pairwise_euclidean_distance

        # self.CTR = CTR(weight_loss_CTR, sinkhorn_alpha, OT_max_iter=sinkhorn_max_iter)

    def make_adam_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer
    

    def make_sam_optimizer(self,):
        base_optimizer = torch.optim.SGD
        if self.SAM_name == 'FSAM':
            optimizer = FSAM(
                self.model.parameters(),
                base_optimizer, device=self.device,
                lr=self.learning_rate,
                sigma=self.sigma, lmbda=self.lmbda
                )
        elif self.SAM_name == 'TRAM':
            optimizer = TRAM(
                self.model.parameters(),
                base_optimizer, device=self.device,
                lr=self.learning_rate,
                sigma=self.sigma, lmbda=self.lmbda
                )
        else:
            print("WRONG!!")
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            lr_scheduler = StepLR(
                optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        else:
            raise NotImplementedError(self.lr_scheduler)
        return lr_scheduler

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
    
        if self.model_name == 'FASTopic':
            train_theta = self.test(dataset_handler.train_contextual_embed, dataset_handler.train_contextual_embed)
        else:
            train_theta = self.test(dataset_handler.train_data)

        return top_words, train_theta

    def train(self, dataset_handler, verbose=False):
        accumulation_steps = self.acc_step
        if self.use_decompose == 1:
            if self.decompose_name  == 'Gram_Schmidt':
                grad_decomposer = Gram_Schmidt(model=self.model, device='cuda', buffer_size=self.task_num)
            elif self.decompose_name == 'SVD':
                grad_decomposer = SVD(model=self.model, device='cuda', buffer_size=self.task_num)
        
        if self.use_MOO != 0:
            if self.MOO_name == 'PCGrad':
                moo_algorithm = PCGrad()
            elif self.MOO_name == 'CAGrad':
                moo_algorithm = CAGrad()
            elif self.MOO_name == 'DB_MTL':
                moo_algorithm = DB_MTL(self.task_num)
            elif self.MOO_name == 'MGDA':
                moo_algorithm = MGDA()
            elif self.MOO_name == 'ExcessMTL':
                moo_algorithm = ExcessMTL(self.task_num)
            elif self.MOO_name == 'FairGrad':
                moo_algorithm = FairGrad()
            elif self.MOO_name == 'MoCo':
                moo_algorithm = MoCo()

        adam_optimizer = self.make_adam_optimizer()
        sam_optimizer = self.make_sam_optimizer() 

        if self.lr_scheduler:
            print("===>using lr_scheduler")
            self.logger.info("===>using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(adam_optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)
        if self.use_SAM == 1:
            print("use SAM")

        print(f"Learn = {self.learn}")
        for epoch_id, epoch in enumerate(tqdm(range(1, self.epochs + 1))):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_id, batch in enumerate(dataset_handler.train_dataloader): 
                *inputs, indices = batch
                batch_data = inputs
                rst_dict = self.model(indices, batch_data, epoch_id=epoch)
                batch_loss = rst_dict['loss_']

                if self.use_SAM == 0:
                    if self.use_MOO == 1:
                        
                        ##
                        batch_loss.backward(retain_graph=True)
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = p.grad.clone()
                        ##

                        sam_optimizer.first_step(zero_grad=True)

                        rst_dict_adv = self.model(indices, batch_data, epoch_id=epoch)
                        loss_sam = rst_dict_adv['loss_']
                        rst_dict['loss_sam'] = loss_sam
                        rst_dict['loss_hieu'] = loss_sam - rst_dict['loss_']

                        if batch_id % 100 == 0:  
                            print(f"Loss: {rst_dict['loss_']}, Loss SAM: {loss_sam}, Difference: {rst_dict['loss_hieu']}")

                        
                        ##
                        total_loss = rst_dict['loss_'] + rst_dict['loss_sam'] + rst_dict['loss_hieu']

                        loss_array = [rst_dict['loss_'], rst_dict['loss_sam'], rst_dict['loss_hieu']]

                        grad_array = [grad_decomposer._get_total_grad(loss) for loss in loss_array]

                        if self.MOO_name == 'MoCo':
                            adjusted_grad, alpha = moo_algorithm.apply(grad_array, loss_array)
                        else:
                            adjusted_grad, alpha = moo_algorithm.apply(grad_array)
                        
                        grad_pointer = 0
                        for p in self.model.parameters():
                            if p.requires_grad:
                                num_params = p.numel()
                                grad_slice = adjusted_grad[grad_pointer:grad_pointer + num_params]
                                p.grad = grad_slice.view_as(p).clone()
                                grad_pointer += num_params

                        encoder_params = list(self.model.encoder1.parameters())
                        encoder_param_ids = set(id(p) for p in encoder_params)
                        
                        other_params = [param for param in self.model.parameters() if id(param) not in encoder_param_ids and param.requires_grad]
                        if other_params:
                            grads = torch.autograd.grad(rst_dict['loss_'], other_params, allow_unused=True)
                            for param, grad in zip(other_params, grads):
                                if grad is not None:
                                    param.grad = grad.clone()

                    # adam_optimizer.step()
                    # adam_optimizer.zero_grad()

                    # batch_loss_adv = rst_dict_adv['loss_']
                        
                        total_loss.backward()

                    sam_optimizer.second_step(zero_grad=True)



                else:
                    if epoch_id > self.epoch_threshold:
                        batch_loss.backward()
                        if self.SAM_name == 'TRAM':
                            self.model.is_CTR = False
                            loss_OT_ = self.model.get_loss_CTR(batch_data, indices)
                            sam_optimizer.first_step(loss_OT_,
                                                    zero_grad=True)
                        else:
                            sam_optimizer.first_step(zero_grad=True)

                        rst_dict_adv = self.model(indices, batch_data, epoch_id=epoch)

                        batch_loss_adv = rst_dict_adv['loss_']
                        batch_loss_adv.backward()

                        sam_optimizer.second_step(zero_grad=True)
                    
                    else:
                        if self.SAM_name == 'TRAM':
                            self.model.is_CTR = True
                        adam_optimizer.zero_grad()
                        batch_loss.backward()
                        adam_optimizer.step()
                    
                for key in rst_dict:
                    try:
                        loss_rst_dict[key] += rst_dict[key] * \
                            len(batch_data['data'])
                    except:
                        loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            if self.lr_scheduler:
                lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                self.logger.info(output_log)


    def test(self, input_data, train_data=None):
        data_size = input_data.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_data[idx]
            
                if self.model_name == 'FASTopic':
                    batch_theta = self.model.get_theta(batch_input, train_data)
                else:
                    batch_theta = self.model.get_theta(batch_input)
                theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta

    def export_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, vocab, num_top_words=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab, num_top_words)
        return top_words

    def export_theta(self, dataset_handler):
        if self.model_name == 'FASTopic':
            train_theta = self.test(dataset_handler.train_contextual_embed, dataset_handler.train_contextual_embed)
            test_theta = self.test(dataset_handler.test_contextual_embed, dataset_handler.train_contextual_embed)
        else:
            train_theta = self.test(dataset_handler.train_data)
            test_theta = self.test(dataset_handler.test_data)
        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f'top_words_{num_top_words}.txt'), 'w') as f:
            for i, words in enumerate(top_words):
                f.write(words + '\n')
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)
        
        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, 'word_embeddings'):
            word_embeddings = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'word_embeddings.npy'), word_embeddings)
            self.logger.info(f'word_embeddings size: {word_embeddings.shape}')

        if hasattr(self.model, 'topic_embeddings'):
            topic_embeddings = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'topic_embeddings.npy'),
                    topic_embeddings)
            self.logger.info(
                f'topic_embeddings size: {topic_embeddings.shape}')

            topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
            np.save(os.path.join(dir_path, 'topic_dist.npy'), topic_dist)

        if hasattr(self.model, 'group_embeddings'):
            group_embeddings = self.model.group_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'group_embeddings.npy'),
                    group_embeddings)
            self.logger.info(
                f'group_embeddings size: {group_embeddings.shape}')

            group_dist = scipy.spatial.distance.cdist(group_embeddings, group_embeddings)
            np.save(os.path.join(dir_path, 'group_dist.npy'), group_dist)

        return word_embeddings, topic_embeddings
