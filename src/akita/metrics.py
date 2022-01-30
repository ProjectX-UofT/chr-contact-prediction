import numpy as np
import torch
import torchmetrics as tm
import torch.nn as nn
import audtorch

def compute_pearson_r_v1(y_pred, y_true):
    y_true = y_true.type(torch.FloatTensor)
    y_pred = y_pred.type(torch.FloatTensor)

    product = torch.sum(torch.multiply(y_true, y_pred))
    true_sum = torch.sum(y_true)
    true_sumsq = torch.sum(torch.square(y_true))
    pred_sum = torch.sum(y_pred)
    pred_sumsq = torch.sum(torch.square(y_pred))
    count = torch.sum(torch.ones_like(y_true))

    true_mean = torch.div(true_sum, count)
    true_mean2 = torch.square(true_mean)
    pred_mean = torch.div(pred_sum, count)
    pred_mean2 = torch.square(pred_mean)

    term1 = product
    term2 = -1 * torch.multiply(true_mean, pred_sum)
    term3 = -1 * torch.multiply(pred_mean, true_sum)
    term4 = torch.multiply(count, torch.multiply(true_mean, pred_mean))
    covariance = term1 + term2 + term3 + term4

    true_var = true_sumsq - torch.multiply(count, true_mean2)
    pred_var = pred_sumsq - torch.multiply(count, pred_mean2)
    pred_var = torch.where(torch.greater(pred_var, 1e-12),
                           pred_var,
                           np.inf * torch.ones_like(pred_var))

    tp_var = torch.multiply(torch.sqrt(true_var), torch.sqrt(pred_var))
    correlation = torch.div(covariance, tp_var)

    return torch.mean(correlation, -1)

def compute_pearson_r_v2(y_pred, y_true):
    return audtorch.metrics.functional.pearsonr(y_pred, y_true)

# def compute_pearson_r_v3(y_pred, y_true):
#     return allennlp.training.metrics.pearson_correlation.PearsonCorrelation.pearson_correlation(y_pred, y_true)

def compute_pearson_r_v4(y_pred, y_true):
    vx = y_pred - torch.mean(y_pred)
    vy = y_true - torch.mean(y_true)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

def compute_pearson_r_v5(y_pred, y_true):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(y_pred - y_pred.mean(dim=1, keepdim=True), y_true - y_true.mean(dim=1, keepdim=True))

# class PearsonR(tm.Metric):
#     def __init__(self, num_targets, summarize=True):
#         super(PearsonR, self).__init__()
#         self._summarize = summarize
#         self._shape = (num_targets,)
#
#         self.add_state(name='_count', default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state(name='_product', default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state(name='_true_sum', default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state(name='_true_sumsq', default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state(name='_pred_sum', default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state(name='_pred_sumsq', default=torch.tensor(0), dist_reduce_fx="sum")
#
#     def update(self, y_pred, y_true):
#         y_true = y_true.type(torch.FloatTensor);
#         y_pred = y_pred.type(torch.FloatTensor);
#
#         product = torch.sum(torch.multiply(y_true, y_pred))
#         self._product.add(product)
#
#         true_sum = torch.sum(y_true)
#         self._true_sum.add(true_sum)
#
#         true_sumsq = torch.sum(torch.square(y_true))
#         self._true_sumsq.add(true_sumsq)
#
#         pred_sum = torch.sum(y_pred)
#         self._pred_sum.add(pred_sum)
#
#         pred_sumsq = torch.sum(torch.square(y_pred))
#         self._pred_sumsq.add(pred_sumsq)
#
#         count = torch.ones_like(y_true)
#         count = torch.sum(count)
#         self._count.add(count)
#
#     def compute(self):
#         true_mean = torch.div(self._true_sum, self._count)
#         true_mean2 = torch.square(true_mean)
#         pred_mean = torch.div(self._pred_sum, self._count)
#         pred_mean2 = torch.square(pred_mean)
#
#         term1 = self._product
#         term2 = -1 * torch.multiply(true_mean, self._pred_sum)
#         term3 = -1 * torch.multiply(pred_mean, self._true_sum)
#         term4 = torch.multiply(self._count, torch.multiply(true_mean, pred_mean))
#         covariance = term1 + term2 + term3 + term4
#
#         true_var = self._true_sumsq - torch.multiply(self._count, true_mean2)
#         pred_var = self._pred_sumsq - torch.multiply(self._count, pred_mean2)
#         pred_var = torch.where(torch.greater(pred_var, 1e-12),
#                                pred_var,
#                                np.inf * torch.ones_like(pred_var))
#
#         tp_var = torch.multiply(torch.sqrt(true_var), torch.sqrt(pred_var))
#         correlation = torch.div(covariance, tp_var)
#
#         if self._summarize:
#             return torch.mean(correlation, -1)
#         else:
#             return correlation
