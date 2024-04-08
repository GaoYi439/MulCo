import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

#################################
# the sifted loss
#################################
class ComSifted(nn.Module):

    def __init__(self, args, base_encoder):
        '''
        :param args: args.num_class: the number of classes
                     args.dim: feature dimension
                     args.arch: ResNet18
                     args.t: softmax temperature (default: 0.07)
                     args.moco_queue: queue size; number of negative keys
                     args.moco_m: moco momentum of updating key encoder (default: 0.999)
        :param base_encoder: SupConResNet model, which adopts ResNet18
        '''

        super(ComSifted, self).__init__()
        pretrained = args.dataset == 'cub200'

        self.m = args.moco_m
        self.T = args.t
        self.K = args.moco_queue

        # create the encoders
        self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.dim, name=args.arch, pretrained=pretrained)
        # momentum encoder
        self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.dim, name=args.arch, pretrained=pretrained)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(args.dim, args.moco_queue))  # dim*moco_queue
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_pre", torch.zeros(args.moco_queue, 1))  # 存储队列元素的预测标签

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            # param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, pre_label):
        '''
        :param keys: k representation
        :param pre_label: the prediction label of img_q (img_q=k)
        :return:
        '''
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        pre_label = concat_all_gather(pre_label)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_pre[ptr:ptr + batch_size, :] = pre_label
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, pre_label):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(pre_label)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, pre_labels, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(pre_labels)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this]

    def forward(self, img_q, img_k=None, com_Y=None, args=None, eval_only=False):

        output, q = self.encoder_q(img_q)  # output: predicted probability (N*num_class); q: queries (N*dim)

        # for testing
        if eval_only:
            return output
        else:
            with torch.no_grad():
                out_k, _ = self.encoder_q(img_k)

        predicetd_scores = torch.softmax(output, dim=1) * (1 - com_Y).clone()  # 保留除补标记外的结果
        _, pre_labels = torch.max(predicetd_scores, dim=1)
        pre_labels = pre_labels.reshape(-1, 1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, pre_labels, idx_unshuffle = self._batch_shuffle_ddp(img_k, pre_labels)

            # out_k, k = self.encoder_k(img_k)  # keys: N*dim
            _, k = self.encoder_k(img_k)  # keys: N*dim


            # undo shuffle
            k, pre_labels = self._batch_unshuffle_ddp(k, pre_labels, idx_unshuffle)



        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1, c refers to dim
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_pos_b = torch.tensor(1.0) / torch.mul(torch.norm(q, dim=1), torch.norm(k, dim=1))
        l_pos_cosin = torch.einsum('nc,n->n', [l_pos, l_pos_b]).unsqueeze(-1)


        # negative logits: Nxmoco_queue, k refers to moco_queue
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        l_neg_b = torch.tensor(1.0)/torch.einsum('nc,ck->nk', [torch.norm(q, dim=1).unsqueeze(-1), torch.norm(self.queue.clone().detach(), dim=0).unsqueeze(0)])
        l_neg_cosin = torch.einsum('nk,nk->nk', [l_neg, l_neg_b])

        logits = torch.cat([l_pos_cosin, l_neg_cosin], dim=1)

        # apply temperature
        logits /= self.T

        # compute exp for N*(1+k)
        exp = torch.exp(logits)
        mask_neg = torch.tensor(1.0) - torch.eq(pre_labels, self.queue_pre.clone().detach().T).float()  # compute different labeled instance with anchor N*k
        mask_pos = torch.ones(exp.shape[0], 1).cuda()
        mask = torch.cat([mask_pos, mask_neg], dim=1)  # N*(1+k)

        # choose the true negative
        sum = torch.tensor(1.0)/torch.sum(exp * mask, dim=1).float()
        true_neg = torch.einsum('nc,n->nc', [exp, sum])  # N*(1+k)
        true_log = torch.log(true_neg)

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, pre_labels)

        return output, true_log, labels, out_k



#################################
# the soft loss
#################################
class ComSoft(nn.Module):

    def __init__(self, args, base_encoder):
        '''
        :param args: args.num_class: the number of classes
                     args.dim: feature dimension
                     args.arch: ResNet18
                     args.t: softmax temperature (default: 0.07)
                     args.moco_queue: queue size; number of negative keys
                     args.moco_m: moco momentum of updating key encoder (default: 0.999)
        :param base_encoder: SupConResNet model, which adopts ResNet18
        '''

        super(ComSoft, self).__init__()
        pretrained = args.dataset == 'cub200'

        self.m = args.moco_m
        self.T = args.t
        self.K = args.moco_queue

        # create the encoders
        self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.dim, name=args.arch, pretrained=pretrained)
        # momentum encoder
        self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.dim, name=args.arch, pretrained=pretrained)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(args.dim, args.moco_queue))  # dim*moco_queue
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_pre", torch.zeros(args.moco_queue, 1))  # 存储队列元素的预测标签
        self.register_buffer("queue_pre_score", torch.zeros(args.num_class, args.moco_queue))  # 存储队列元素的预测值

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            # param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, pre_label, predicetd_scores):
        '''
        :param keys: k representation
        :param pre_label: the prediction label of img_q (img_q=k)
        :return:
        '''
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        pre_label = concat_all_gather(pre_label)
        predicetd_scores = concat_all_gather(predicetd_scores)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_pre[ptr:ptr + batch_size, :] = pre_label
        self.queue_pre_score[:, ptr:ptr + batch_size] = predicetd_scores.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, pre_label, predicetd_scores):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(pre_label)
        score_gather = concat_all_gather(predicetd_scores)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], score_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, pre_labels, predicetd_scores, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(pre_labels)
        score_gather = concat_all_gather(predicetd_scores)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], score_gather[idx_this]

    def forward(self, img_q, img_k=None, com_Y=None, args=None, eval_only=False):

        output, q = self.encoder_q(img_q)  # output: predicted probability (N*num_class); q: queries (N*dim)

        # for testing
        if eval_only:
            return output
        else:
            with torch.no_grad():
                out_k, _ = self.encoder_q(img_k)

        predicetd_scores = torch.softmax(output, dim=1) * (1 - com_Y).clone()  # 保留除补标记外的结果
        _, pre_labels = torch.max(predicetd_scores, dim=1)
        pre_labels = pre_labels.reshape(-1, 1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, pre_labels, predicetd_scores, idx_unshuffle = self._batch_shuffle_ddp(img_k, pre_labels, predicetd_scores)

            _, k = self.encoder_k(img_k)  # keys: N*dim


            # undo shuffle
            k, pre_labels, predicetd_scores = self._batch_unshuffle_ddp(k, pre_labels, predicetd_scores, idx_unshuffle)


        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1, c refers to dim
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # negative logits: Nxmoco_queue, k refers to moco_queue
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # apply temperature
        l_pos /= self.T
        l_neg /= self.T

        # compute exp for positive and negatives
        exp_pos = torch.exp(l_pos)
        exp_neg = torch.exp(l_neg)

        # weight to balance the positive and negative
        scores = self.queue_pre_score.clone().detach()
        weight = scores[pre_labels.squeeze(-1), :]  # N * moco_queue 选择所有负样本在正样本上的预测为权重
        l_neg_weight = torch.einsum('nk,nk->nk', [(1.0000001-weight).clone(), exp_neg])

        exp = torch.cat([exp_pos, l_neg_weight], dim=1)  # N*(k+1)
        logits = l_pos - torch.log(torch.sum(exp, dim=1, keepdim=True) + 1e-12)

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, pre_labels, predicetd_scores)

        return output, logits, labels, out_k



#################################
# the weighted loss
#################################
class ComWeight(nn.Module):

    def __init__(self, args, base_encoder):
        '''
        :param args: args.num_class: the number of classes
                     args.dim: feature dimension
                     args.arch: ResNet18
                     args.t: softmax temperature (default: 0.07)
                     args.moco_queue: queue size; number of negative keys
                     args.moco_m: moco momentum of updating key encoder (default: 0.999)
        :param base_encoder: SupConResNet model, which adopts ResNet18
        '''

        super(ComWeight, self).__init__()
        pretrained = args.dataset == 'cub200'

        self.m = args.moco_m
        self.T = args.t
        self.K = args.moco_queue

        # create the encoders
        self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.dim, name=args.arch, pretrained=pretrained)
        # momentum encoder
        self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.dim, name=args.arch, pretrained=pretrained)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(args.dim, args.moco_queue))  # dim*moco_queue
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_pre_score", torch.zeros(args.num_class, args.moco_queue))  # 存储队列元素的预测值

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            # param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, predicetd_scores):
        '''
        :param keys: k representation
        :param pre_label: the prediction label of img_q (img_q=k)
        :return:
        '''
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        predicetd_scores = concat_all_gather(predicetd_scores)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_pre_score[:, ptr:ptr + batch_size] = predicetd_scores.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, predicetd_scores):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        score_gather = concat_all_gather(predicetd_scores)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], score_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, predicetd_scores, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        score_gather = concat_all_gather(predicetd_scores)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], score_gather[idx_this]

    def forward(self, img_q, img_k=None, com_Y=None, args=None, eval_only=False):

        output, q = self.encoder_q(img_q)  # output: predicted probability (N*num_class); q: queries (N*dim)

        # for testing
        if eval_only:
            return output
        else:
            with torch.no_grad():
                out_k, _ = self.encoder_q(img_k)

        predicetd_scores = torch.softmax(output, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, predicetd_scores, idx_unshuffle = self._batch_shuffle_ddp(img_k, predicetd_scores)

            # out_k, k = self.encoder_k(img_k)  # keys: N*dim
            _, k = self.encoder_k(img_k)  # keys: N*dim


            # undo shuffle
            k, predicetd_scores = self._batch_unshuffle_ddp(k, predicetd_scores, idx_unshuffle)


        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1, c refers to dim
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)


        # negative logits: Nxmoco_queue, k refers to moco_queue
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # apply temperature
        l_pos /= self.T
        l_neg /= self.T

        # compute exp for positive and negatives
        exp_pos = torch.exp(l_pos)
        exp_neg = torch.exp(l_neg)

        # weight to balance the positive and negative
        com_labels = com_Y.clone().float()
        weight = torch.einsum('nc,ck->nk', [com_labels, self.queue_pre_score.clone().detach()])  # N * moco_queue 对于负样本，选择正样本补标记位置相对应的预测概率之和作为权重
        l_neg_weight = torch.einsum('nk,nk->nk', [weight.clone(), exp_neg])

        exp = torch.cat([exp_pos, l_neg_weight], dim=1)
        logits = l_pos - torch.log(torch.sum(exp, dim=1, keepdim=True) + 1e-12)

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, predicetd_scores)
        return output, logits, labels, out_k