import numpy as np
import mxnet as mx

class AccMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(AccMetric, self).__init__(
        'acc', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    self.count+=1
    label = labels[0]
    pred_label = preds[1]
    #print('ACC', label.shape, pred_label.shape)
    if pred_label.shape != label.shape:
        pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
    pred_label = pred_label.asnumpy().astype('int32').flatten()
    label = label.asnumpy()
    if label.ndim==2:
      label = label[:,0]
    label = label.astype('int32').flatten()
    assert label.shape==pred_label.shape
    self.sum_metric += (pred_label.flat == label.flat).sum()
    self.num_inst += len(pred_label.flat)
## preds[0] classier loss
class LossValueMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossValueMetric, self).__init__(
        'lossvalue', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
    #label = labels[0].asnumpy()
    # print("^^^^^^^^^^^^len(preds) ^^^^^^^^^^^^^^^^^")
    # print (len(preds))
    # # ((16, 512), (16, 97768), (1,), (16, 1))  this is symbo return outlist include 
    # # embedding softmax ce_loss  me_add_loss
    # print (preds[0].shape,preds[1].shape,preds[2].shape)#,preds[3].shape)
    pred = preds[-1].asnumpy()  #celoss
    #print('in loss', pred.shape)
    #print(pred)
    loss = pred[0]
    self.sum_metric += loss
    self.num_inst += 1.0
    #gt_label = preds[-2].asnumpy()
    #print(gt_label)
class AccMetric_d(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(AccMetric_d, self).__init__(
        'acc', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    self.count+=1
    label = labels[0]
    pred_label = preds[0]
    #print('ACC', label.shape, pred_label.shape)
    if pred_label.shape != label.shape:
        pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
    pred_label = pred_label.asnumpy().astype('int32').flatten()
    label = label.asnumpy()
    if label.ndim==2:
      label = label[:,0]
    label = label.astype('int32').flatten()
    assert label.shape==pred_label.shape
    self.sum_metric += (pred_label.flat == label.flat).sum()
    self.num_inst += len(pred_label.flat)

class LossValueMetric_d(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossValueMetric_d, self).__init__(
        'lossvalue', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
    #label = labels[0].asnumpy()
    pred = preds[-1].asnumpy()
    #print('in loss', pred.shape)
    #print(pred)
    loss = pred[0]
    self.sum_metric += loss
    self.num_inst += 1.0
    #gt_label = preds[-2].asnumpy()
    #print(gt_label)




