#_*_coding:utf-8_*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import sklearn
import pickle
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
from config import config, default, generate_config
from metric import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
# import flops_counter
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
import verification
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbol'))
import fresnet
import fmobilefacenet
import fmobilenet
import fmnasnet
import fdensenet
import fresnet_sge
import cv2



logger = logging.getLogger()
logger.setLevel(logging.INFO)


args = None



def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--dataset', default=default.dataset, help='dataset config')
  parser.add_argument('--network', default=default.network, help='network config')
  parser.add_argument('--loss', default=default.loss, help='loss config')
  args, rest = parser.parse_known_args()
  generate_config(args.network, args.dataset, args.loss)
  parser.add_argument('--models-root', default=default.models_root, help='root directory to save model.')
  parser.add_argument('--pretrained', default=default.pretrained, help='pretrained model to load')
  parser.add_argument('--pretrained-epoch', type=int, default=default.pretrained_epoch, help='pretrained epoch to load')
  parser.add_argument('--ckpt', type=int, default=default.ckpt, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--verbose', type=int, default=default.verbose, help='do verification testing and model saving every verbose batches')
  parser.add_argument('--lr', type=float, default=default.lr, help='start learning rate')
  parser.add_argument('--lr-steps', type=str, default=default.lr_steps, help='steps of lr changing')
  parser.add_argument('--wd', type=float, default=default.wd, help='weight decay')
  parser.add_argument('--mom', type=float, default=default.mom, help='momentum')
  parser.add_argument('--frequent', type=int, default=default.frequent, help='')
  parser.add_argument('--per-batch-size', type=int, default=default.per_batch_size, help='batch size in each context')
  parser.add_argument('--kvstore', type=str, default=default.kvstore, help='kvstore setting')
  args = parser.parse_args()
  return args
### me top test
class Embedding:
    def __init__(self, prefix, epoch, ctx_id=0):
        print('loading', prefix, epoch)
        ctx = mx.gpu(ctx_id)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        image_size = (112, 112)
        self.image_size = image_size
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def get(self, rimg):
        img = rimg#cv2.imread()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        # img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((1, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        # input_blob[1] = img_flip
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        feat = self.model.get_outputs()[0].asnumpy()
        feat = feat.reshape([-1, feat.shape[0]])#* feat.shape[1]]) #512 shape
        feat = feat.flatten()
        return feat
        
def get_image_feature(img_path, img_list_path, model_path, gpu_idd,mbatch):
    img_list = open(img_list_path)
    embedding = Embedding(model_path, mbatch, gpu_idd)
    files = img_list.readlines()[0:5000]
    img_feats = []
    for img_index, each_line in enumerate(((files))):
        img_name = os.path.join(img_path, each_line.strip().split()[0])
        img = cv2.imread(img_name)
        if img.shape[0]!=112 and img.shape[1]!=112:
            img=cv2.resize(img, (30, 30),interpolation=cv2.INTER_CUBIC) ##������
            img=cv2.resize(img, (112, 112),interpolation=cv2.INTER_CUBIC)
        img_feats.append(embedding.get(img))
    img_feats = np.array(img_feats).astype(np.float32)
    return img_feats
def my_top(epoch):
    img_path = '/home/svt/mxnet_recognition/dataes/30size' # img path
    img_list_path = '/home/svt/mxnet_recognition/dataes/30size_1761.txt'  #img txt path
    model_path = "/home/svt/mxnet_recognition/modify_model_output/gan_transfer_IR-sger50-arcface/model/sger50-arcface-emore/model"
    # model_path = "/home/svt/mxnet_recognition/modify_model_output/IR_r50_low_resolution-sger50-arcface/model/sger50-arcface-emore/modelfc7"
    gpu_idd = 0  #����train.sh ��GPU �豸�����ǻ������豸
    epoch = epoch
    img_feats30 = get_image_feature(img_path, img_list_path, model_path, gpu_idd,epoch)
    
    
    img_path = '/home/svt/mxnet_recognition/dataes/Set3' # img path
    img_list_path = '/home/svt/mxnet_recognition/dataes/surve_test_set3.txt'  #img txt path
    suever_img_feats = get_image_feature(img_path, img_list_path, model_path, gpu_idd,epoch)
    t_feats = suever_img_feats[:,:] ##img_feats[0:780,:]
    g_feats = np.concatenate((suever_img_feats[:,:],img_feats30[:,:]),axis=0)#gallery_label[:, 0:gallery_label.shape[1] / 2]
    print("*************************")
    print ("test number",t_feats.shape)
    print ("gallery number",g_feats.shape)
    g_feats = g_feats / np.sqrt(np.sum(g_feats ** 2, -1, keepdims=True))
    t_feats = t_feats / np.sqrt(np.sum(t_feats ** 2, -1, keepdims=True))

    label=[]  #test label
    f=open("/home/svt/mxnet_recognition/dataes/surve_test_set3.txt")
    labels = f.readlines()
    f.close()
    for l in labels:
        l=l.strip().split()
        label.append(l[1])
    # gallery label
    f=open('/home/svt/mxnet_recognition/dataes/30size_1761.txt')
    labels = f.readlines()[0:5000]
    f.close()
    for l in labels:
        l=l.strip().split()
        label.append(l[2])
    correct1 = 0
    correct10 = 0
    print(len(label))
    for i,line in enumerate(t_feats):
        line = np.tile(line,(len(g_feats),1))  # repeat gallery number
        dis = np.sum(g_feats * line, 1)  # save index  correspond index
        sort_index = np.argsort(-dis, axis=0) #this is label message,  small to large

        top100=[]
        temp_label=[]
        # for j in range(101): #get sort index
        for j in range(11): #get sort index
            # top100.append(dis[sort_index[j]])
            temp_label.append(label[sort_index[j]])  ##gallery label
                
        if label[i] in temp_label[1:11]:   #test_label
            correct10=correct10+1
        if label[i] in temp_label[1:2]:
            correct1=correct1+1

    print ("survers img top10 is : ", correct10 / float(len(t_feats)))
    print ("survers img top1 is : ", correct1 / float(len(t_feats)))
    return correct1 / float(len(t_feats)),correct10 / float(len(t_feats))
    
def my_top_yidong_test(epoch):
    img_path = '/home/svt/mxnet_recognition/dataes/30size' # img path
    img_list_path = '/home/svt/mxnet_recognition/dataes/30size_1761.txt'  #img txt path
    model_path = "/home/svt/mxnet_recognition/modify_model_output/gan_transfer_IR-sger50-arcface/model/sger50-arcface-emore/model"
    gpu_idd = 0  #����train.sh ��GPU �豸�����ǻ������豸
    epoch = epoch
    img_feats30 = get_image_feature(img_path, img_list_path, model_path, gpu_idd,epoch)
    img_feats=img_feats30
    t_feats = img_feats[0:780,:]
    g_feats = img_feats[:,:]#gallery_label[:, 0:gallery_label.shape[1] / 2]
    print("*************************")
    print ("test number",t_feats.shape)
    print ("gallery number",g_feats.shape)
    g_feats = g_feats / np.sqrt(np.sum(g_feats ** 2, -1, keepdims=True))
    t_feats = t_feats / np.sqrt(np.sum(t_feats ** 2, -1, keepdims=True))

    label=[]
    f=open('/home/svt/mxnet_recognition/dataes/30size_1761.txt')
    labels = f.readlines()[0:5000]
    f.close()
    for l in labels:
        l=l.strip().split()
        label.append(l[2])
    correct1 = 0
    correct10 = 0
    for i,line in enumerate(t_feats):
        line = np.tile(line,(len(g_feats),1))  # repeat gallery number
        dis = np.sum(g_feats * line, 1)  # save index  correspond index
        sort_index = np.argsort(-dis, axis=0) #this is label message,  small to large

        top100=[]
        temp_label=[]
        # for j in range(101): #get sort index
        for j in range(11): #get sort index
            # top100.append(dis[sort_index[j]])
            temp_label.append(label[sort_index[j]])  ##gallery label
                
        if label[i] in temp_label[1:11]:   #test_label
            correct10=correct10+1
        if label[i] in temp_label[1:2]:
            correct1=correct1+1

    print ("30resize top10 is : ", correct10 / float(len(t_feats)))
    print ("30resize top1 is : ", correct1 / float(len(t_feats)))
    return correct1 / float(len(t_feats)),correct10 / float(len(t_feats))


def get_symbol(args):
  #network.r100.net_name = 'fresnet', eval('fresnet').get_symbol() ==  fresnet.get_symbol()
  
  
  embedding = eval(config.net_name).get_symbol()    #fresnet.py ����data = mx.symbol.Variable('data')
  all_label = mx.symbol.Variable('softmax_label')  #ģ�ͽṹ���ñ�ǩ��������Ҫ��������ʧ

  gt_label = all_label
  is_softmax = True
  if config.loss_name=='softmax': #softmax
    _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size), 
        lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
    if config.fc7_no_bias:
      fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
    else:
      _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
      fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=config.num_classes, name='fc7')
  elif config.loss_name=='margin_softmax':  # arcface loss
    _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size), ##512
        lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
    s = config.loss_s
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')

    if config.loss_m1!=1.0 or config.loss_m2!=0.0 or config.loss_m3!=0.0:
      if config.loss_m1==1.0 and config.loss_m2==0.0:
        s_m = s*config.loss_m3
        gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = s_m, off_value = 0.0)
        fc7 = fc7-gt_one_hot  #fc7��������Ԥ�����onehot
      else:
        zy = mx.sym.pick(fc7, gt_label, axis=1)
        cos_t = zy/s
        t = mx.sym.arccos(cos_t)
        if config.loss_m1!=1.0:
          t = t*config.loss_m1
        if config.loss_m2>0.0:
          t = t+config.loss_m2
        body = mx.sym.cos(t)
        if config.loss_m3>0.0:
          body = body - config.loss_m3
        new_zy = body*s
        diff = new_zy - zy
        diff = mx.sym.expand_dims(diff, 1)
        gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 1.0, off_value = 0.0)
        body = mx.sym.broadcast_mul(gt_one_hot, diff)
        fc7 = fc7+body
  elif config.loss_name.find('triplet')>=0:
    is_softmax = False
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
    anchor = mx.symbol.slice_axis(nembedding, axis=0, begin=0, end=args.per_batch_size//3)
    positive = mx.symbol.slice_axis(nembedding, axis=0, begin=args.per_batch_size//3, end=2*args.per_batch_size//3)
    negative = mx.symbol.slice_axis(nembedding, axis=0, begin=2*args.per_batch_size//3, end=args.per_batch_size)
    if config.loss_name=='triplet':
      ap = anchor - positive
      an = anchor - negative
      ap = ap*ap
      an = an*an
      ap = mx.symbol.sum(ap, axis=1, keepdims=1) #(T,1)
      an = mx.symbol.sum(an, axis=1, keepdims=1) #(T,1)
      triplet_loss = mx.symbol.Activation(data = (ap-an+config.triplet_alpha), act_type='relu')
      triplet_loss = mx.symbol.mean(triplet_loss)
    else:
      ap = anchor*positive
      an = anchor*negative
      ap = mx.symbol.sum(ap, axis=1, keepdims=1) #(T,1)
      an = mx.symbol.sum(an, axis=1, keepdims=1) #(T,1)
      ap = mx.sym.arccos(ap)
      an = mx.sym.arccos(an)
      triplet_loss = mx.symbol.Activation(data = (ap-an+config.triplet_alpha), act_type='relu')
      triplet_loss = mx.symbol.mean(triplet_loss)
    triplet_loss = mx.symbol.MakeLoss(triplet_loss)
  out_list = [mx.symbol.BlockGrad(embedding)]
  if is_softmax:
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax', normalization='valid')
    out_list.append(softmax)
    if config.ce_loss:  #Cross Entropy Function  is ce
      #ce_loss = mx.symbol.softmax_cross_entropy(data=fc7, label = gt_label, name='ce_loss')/args.per_batch_size
      body = mx.symbol.SoftmaxActivation(data=fc7)
      body = mx.symbol.log(body)
      _label = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = -1.0, off_value = 0.0)

      body = body*_label

      ce_loss = mx.symbol.sum(body)/args.per_batch_size
      out_list.append(mx.symbol.BlockGrad(ce_loss))
  else:
    out_list.append(mx.sym.BlockGrad(gt_label))
    out_list.append(triplet_loss)
    
  #��������������Ǽ������Ķ������ǩԤ��ֵ����ʵ�ʵı�ǩԤ��ֵ
  # gan_label = mx.symbol.Variable('gan_label')  #nchw L2
  # gan_loss = mx.symbol.softmax_cross_entropy(data=fc1, label=gt_label, name='ce_loss') / args.per_batch_size
  #
  #
  # t_feature=mx.symbol.L2Normalization(teacher, name='l2_norm_high')
  # s_feature=mx.symbol.L2Normalization(embedding, name='l2_norm_low')
  # pred = mx.sym.sqrt(mx.sym.sum(mx.sym.square(t_feature - s_feature), axis=1, keepdims=True)) #
  # contrative_loss = mx.sym.MakeLoss(pred, name='loss')
  # out_list.append(contrative_loss)
  # out = mx.symbol.Group(out_list)


  #add discrimi model and ganloss
  return out

##mxnet
def discriminator(args): ##�������δ�������ݲ�һ�������Ǻϲ������ݺͺϲ��ı�ǩ��������ʧ��Ȼ��ͬ��������ֻ��ѧ����������ݺͱ�ǩ����ʧ
    data = mx.sym.Variable(name='data') # teache 512 concat stuedent 512,,is (2*batch 512)
    label = mx.symbol.Variable('softmax_label') #label is  (2*batch 2) 0 and 1,teach is 1 student is 0
    fc1 = mx.sym.FullyConnected(data=data, num_hidden=512,name="fc1") #512 this is me add ,can set any
    act1=mx.sym.LeakyReLU(data=fc1, act_type='prelu', name="prelu1")
    fc2 = mx.sym.FullyConnected(data=act1, num_hidden=512, name="fc2")  # 512 this is me add ,can set any
    act2 = mx.sym.LeakyReLU(data=fc2, act_type='prelu', name="prelu2")
    fc3 = mx.sym.FullyConnected(data=act2, num_hidden=2, name="fc3")  # class loss

    softmax = mx.symbol.SoftmaxOutput(data=fc3, label = label, name='softmax', normalization='valid')
    #ce_loss = mx.symbol.softmax_cross_entropy(data=fc7, label = gt_label, name='ce_loss')/args.per_batch_size
    body = mx.symbol.SoftmaxActivation(data=fc3)
    body = mx.symbol.log(body)
    config.num_classes=2
    _label = mx.sym.one_hot(label, depth = config.num_classes, on_value = -1.0, off_value = 0.0)

    body = body*_label

    ce_loss = mx.symbol.sum(body)/args.per_batch_size
    return mx.symbol.Group([softmax,mx.symbol.BlockGrad(ce_loss)])


def two_sym(args):
    ### adda gan loss ,����Ҫ����ṹ����Ԥѵ���ý�ʦ�߷ֱ������磬Ȼ��ͨ����������ʧ�����Ż�Ŀ�����磬�ҵ����ߵĹ�ͬ��
    ### �������ݣ�Դ���ݣ���Ŀ�����ݣ�Դmodel �õ���srm,Ŀ�������õ��� IRres
    #####  sym, arg_params, aux_params = mx.model.load_checkpoint
    print('loading', args.pretrained, args.pretrained_epoch)  # ����Ԥѵ��ģ��
    # _, arg_params, aux_params = mx.model.load_checkpoint(args.pretrained, args.pretrained_epoch)

    sym, arg_params, aux_params = mx.model.load_checkpoint(args.pretrained, args.pretrained_epoch)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#################################")
    # sym = get_symbol(args)  ##ģ�ͽṹ ,���fc7 �ĳ������������д��
    print('loading high resoluton model')  # ����Ԥѵ��ģ��
    # sym2, arg_params, aux_params = mx.model.load_checkpoint(args.pretrained, args.pretrained_epoch)
    srm_model_path = "./glint_srm_modelfc7"
    srm_epoch =12
    sym_high, t_arg_params, t_aux_params = mx.model.load_checkpoint(srm_model_path , srm_epoch)
    # sym_high = get_symbol(args)  #��ʦ����ֻ���ص� fc1 �����
    all_layers = sym_high.get_internals()
    sym_high = all_layers['fc1_output']  #param can more ,but load auto to fc1 param
    # sym_high_l2 = mx.symbol.L2Normalization(data=sym_high, name='l2_norm_high')
    print("��������ģ���������")
    return sym,sym_high,arg_params,aux_params,t_arg_params, t_aux_params

def train_net(args):
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
      for i in xrange(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      print('use cpu')
    else:
      print('gpu num:', len(ctx))
    prefix = os.path.join(args.models_root, '%s-%s-%s'%(args.network, args.loss, args.dataset), 'model')
    prefix_dir = os.path.dirname(prefix)
    print('prefix', prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    args.ctx_num = len(ctx)  #GPU num
    args.batch_size = args.per_batch_size*args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = config.image_shape[2]
    config.batch_size = args.batch_size
    config.per_batch_size = args.per_batch_size

    data_dir = config.dataset_path
    path_imgrec = None
    path_imglist = None
    image_size = config.image_shape[0:2]
    assert len(image_size)==2
    assert image_size[0]==image_size[1]
    print('image_size', image_size)
    print('num_classes', config.num_classes)
    path_imgrec = os.path.join(data_dir, "train.rec")

    print('Called with argument:', args, config)
    data_shape = (args.image_channel,image_size[0],image_size[1]) # chw
    mean = None #[127.5,127.5,127.5]
    


    begin_epoch = 0
    if len(args.pretrained)==0:
      arg_params = None
      aux_params = None
      sym = get_symbol(args)  
      if config.net_name=='spherenet':
        data_shape_dict = {'data' : (args.per_batch_size,)+data_shape}
        spherenet.init_weights(sym, data_shape_dict, args.num_layers)
    else:  #��Ԥѵ��ģ�ͣ�������,sym����get_symbol(args)������

      sym,sym_high,arg_params,aux_params,t_arg_params, t_aux_params = two_sym(args)
      d_sym = discriminator(args)

      
            
    config.count_flops=False #me add
    if config.count_flops:  #true
      all_layers = sym.get_internals()
      _sym = all_layers['fc1_output']  #ͼƬ�� 128 ά�ȵ�����fc1 ���ٶ�
      FLOPs = flops_counter.count_flops(_sym, data=(1,3,image_size[0],image_size[1]))
      _str = flops_counter.flops_str(FLOPs)
      print('Network FLOPs: %s'%_str)

    #label_name = 'softmax_label'
    #label_shape = (args.batch_size,)

    val_dataiter = None

    if config.loss_name.find('triplet')>=0:
      from triplet_image_iter import FaceImageIter
      triplet_params = [config.triplet_bag_size, config.triplet_alpha, config.triplet_max_ap]
      train_dataiter = FaceImageIter(
          batch_size           = args.batch_size,
          data_shape           = data_shape,
          path_imgrec          = path_imgrec,
          shuffle              = True,
          rand_mirror          = config.data_rand_mirror,
        #   rand_resize          = True, #me add to differ resolution img 
          mean                 = mean,
          cutoff               = config.data_cutoff,
          ctx_num              = args.ctx_num,
          images_per_identity  = config.images_per_identity,
          triplet_params       = triplet_params,
          mx_model             = model,
      )
      _metric = LossValueMetric()
      eval_metrics = [mx.metric.create(_metric)]
    else:
      from distribute_image_iter import FaceImageIter

      train_dataiter_low = FaceImageIter(  #�õ� batch  img  label, train_dataiter_high
          batch_size           = args.batch_size,
          data_shape           = data_shape,
          path_imgrec          = path_imgrec,
          shuffle              = True,
          rand_mirror          = config.data_rand_mirror, #true
          rand_resize          = True, #me add to differ resolution img 
          mean                 = mean,
          cutoff               = config.data_cutoff,  #0
          color_jittering      = config.data_color,  #0
          images_filter        = config.data_images_filter, #0
      )
      source_imgrec = os.path.join("/home/svt/mxnet_recognition/dataes/faces_glintasia","train.rec")
      data2 = FaceImageIter(  #�õ� batch  img  label, train_dataiter_high
          batch_size           = args.batch_size,
          data_shape           = data_shape,
          path_imgrec          = source_imgrec,
          shuffle              = True,
          rand_mirror          = config.data_rand_mirror, #true
          rand_resize          = False, #me add to differ resolution img
          mean                 = mean,
          cutoff               = config.data_cutoff,  #0
          color_jittering      = config.data_color,  #0
          images_filter        = config.data_images_filter, #0
      )
      metric1 = AccMetric()  #�õ����ȼ���
      eval_metrics = [mx.metric.create(metric1)]
      if config.ce_loss:  #is True
        metric2 = LossValueMetric()  #�õ���ʧֵ
        eval_metrics.append( mx.metric.create(metric2) )  #

    if config.net_name=='fresnet' or config.net_name=='fmobilefacenet':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    #initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    _rescale = 1.0/args.ctx_num
    #opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd, rescale_grad=_rescale)
    opt = optimizer.Adam(learning_rate=0.0001, beta1=0.5, beta2=0.9, epsilon=1e-08)
    _cb = mx.callback.Speedometer(args.batch_size, args.frequent)

    ver_list = []
    ver_name_list = []
    for name in config.val_targets:
      path = os.path.join(data_dir,name+".bin")
      if os.path.exists(path):
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)
        print('ver', name)



    def ver_test(nbatch):
      results = []
      for i in xrange(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, 10, None, None)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
      return results



    highest_acc = [0.0, 0.0]  #lfw and target
    #for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    high_save = 0 #  me  add
    print('lr_steps', lr_steps)
    def _batch_callback(param):
      #global global_step
      global_step[0]+=1
      mbatch = global_step[0]
      for step in lr_steps:
        if mbatch==step:
          opt.lr *= 0.1
          print('lr change to', opt.lr)
          break

      _cb(param)
      if mbatch%1000==0:
        print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)
      
      if mbatch %4000==0:#(fc7_save):
          name=os.path.join(args.models_root, '%s-%s-%s'%(args.network, args.loss, args.dataset), 'modelfc7')
          arg, aux = model.get_params()
          mx.model.save_checkpoint(name, param.epoch, model.symbol, arg, aux)
          print('save model include fc7 layer')
          print("mbatch",mbatch)
      
      me_msave=0
      if mbatch>=0 and mbatch%args.verbose==0:  #default.verbose = 2000,mbatch is
        acc_list = ver_test(mbatch)
        save_step[0]+=1
        msave = save_step[0]  # batch ��512��һ��epoch1300
        me_msave=me_msave+1
        do_save = False
        is_highest = False
        #me add
        save2 = False
        if len(acc_list)>0:
          lfw_score = acc_list[0]
          if lfw_score>highest_acc[0]:
            highest_acc[0] = lfw_score
            if lfw_score>=0.9960:
              save2 = True
              
          score = sum(acc_list)
          if acc_list[-1]>=highest_acc[-1]:
            if acc_list[-1]>highest_acc[-1]:
              is_highest = True
            else:
              if score>=highest_acc[0]:
                is_highest = True
                highest_acc[0] = score
            highest_acc[-1] = acc_list[-1]
            #if lfw_score>=0.99:
            #  do_save = True
        # if is_highest:
          # do_save = True
        if args.ckpt==0:
          do_save = False
        elif args.ckpt==2:
          do_save = True
        elif args.ckpt==3 and is_highest:  #me add and is_highest
          high_save = 0   #ÿ�α���lfw��ߵ�ģ��,�и��ߵ��滻ԭ�������ģ��

        if do_save:  #������ߵ����ݲ���
          print('saving high pretrained-epoch always:  ', high_save)
          arg, aux = model.get_params()
          if config.ckpt_embedding:  #true
            all_layers = model.symbol.get_internals()
            _sym = all_layers['fc1_output']
            _arg = {}
            for k in arg:
              if not k.startswith('fc7'):#�ַ�����ʼ�� fc7 ��ͷ������ѭ�������������������㣩
                _arg[k] = arg[k]
            mx.model.save_checkpoint(prefix, high_save, _sym, _arg, aux)  #��������֣������ǰ׺������Ĳ���ֻ��fc1(128ά�ȵ�����)
          else:
            mx.model.save_checkpoint(prefix, high_save, model.symbol, arg, aux)
          print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
          
        if save2:
          arg, aux = model.get_params()
          if config.ckpt_embedding:  #true
            all_layers = model.symbol.get_internals()
            _sym = all_layers['fc1_output']
            _arg = {}
            for k in arg:
              if not k.startswith('fc7'):#�ַ�����ʼ�� fc7 ��ͷ������ѭ�������������������㣩
                _arg[k] = arg[k]
            mx.model.save_checkpoint(prefix, (me_msave), _sym, _arg, aux)  #��������֣������ǰ׺������Ĳ���ֻ��fc1(128ά�ȵ�����)
          else:
            mx.model.save_checkpoint(prefix, (me_msave), model.symbol, arg, aux)
          print("save pretrained-epoch :param.epoch + me_msave",param.epoch,me_msave)
          print('[%d]LFW Accuracy>=0.9960: %1.5f'%(mbatch, highest_acc[-1])) #mbatch  �Ǵ�0 ��13000 һ��epoch ,Ȼ���ٴ�0����
    
      if config.max_steps>0 and mbatch>config.max_steps:
        sys.exit(0)
        
    ###########################################################################
   
    
    
    epoch_cb = None
    train_dataiter_low = mx.io.PrefetchingIter(train_dataiter_low) #���̵߳�����
    data2 = mx.io.PrefetchingIter(data2)  # ���̵߳�����

    #����model, �õ����ݣ�bind(data��label,�������ִ�к󣬷�����Դ�ռ�)��Ȼ���ʼ���������params
    #Ȼ�� fit ����ѵ��
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[100, 200, 300], factor=0.1)
    optimizer_params = {'learning_rate':0.01,
                    'momentum':0.9,
                    'wd':0.0005,
                    # 'lr_scheduler':lr_scheduler,
                    "rescale_grad":_rescale}  #���ݶȽ�����ƽ��
    ######################################################################
    # # ��ʦ����
    data_shapes = [('data', (args.batch_size, 3, 112, 112))]  #teacher model only need data, no label 
    t_module = mx.module.Module(symbol=sym_high, context=ctx, label_names=[])
    t_module.bind(data_shapes=data_shapes, for_training=False, grad_req='null')
    t_module.set_params(arg_params=t_arg_params, aux_params=t_aux_params)
    t_model=t_module
    ######################################################################
    ##ѧ������
    label_shapes = [('softmax_label', (args.batch_size, ))]
    model = mx.mod.Module(
    context       = ctx,
    symbol        = sym,
    label_names=[]
    # data_names    =  #Ĭ��data,�� softmax_label,����Ķ���label �����֣���Ҫ���´���
    )
    #ѧ��������Ҫ ���ݺͱ�ǩ����ѵ��
    #��ʦ������Ҫ���ݣ����ñ�ǩ����ѵ�������Ұ����������ֵ��ӵ���ǩ����
    # print (train_dataiter_low.provide_data)
    # print ((train_dataiter_low.provide_label))
    #opt_d = optimizer.SGD(learning_rate=args.lr*0.01, momentum=args.mom, wd=args.wd, rescale_grad=_rescale) ##lr e-5
    opt_d = optimizer.Adam(learning_rate=0.0001, beta1=0.5, beta2=0.9, epsilon=1e-08)
    model.bind(data_shapes=data_shapes,for_training=True) #label shape���ˣ����˱�ǩ��������
    model.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=True)  #���Ϊtrue����������ܰ���ȱ�ٵ�ֵ�����ҽ����ó�ʼֵ�趨���������Щȱ�ٵĲ���
    # model.init_optimizer(kvstore=args.kvstore,optimizer='sgd', optimizer_params=(optimizer_params))
    model.init_optimizer(kvstore=args.kvstore,optimizer=opt_d)
    # metric = eval_metrics  #�������㣬�б�
    ##########################################################################
    ## ����������
    # ����ģ�飬�Ǳ����
    model_d = mx.module.Module(symbol=d_sym, context=ctx,data_names=['data'], label_names=['softmax_label'])
    data_shapes = [('data', (args.batch_size*2,512))]
    label_shapes = [('softmax_label', (args.batch_size*2,))]  #bind ������Զ��ı�batch��С��Ҳ����ʹ�õ�ʱ���ٰ�
    model_d.bind(data_shapes=data_shapes,label_shapes = label_shapes,inputs_need_grad=True)
    model_d.init_params(initializer=initializer)
    model_d.init_optimizer(kvstore=args.kvstore,optimizer=opt) #�Ż���������Ҫ�Ķ� #lr e-3
    ## �����õ��ǣ������� discriminator  �������������
    metric_d = AccMetric_d()  #�õ����ȼ���,��metric.py ��Ӻ���AccMetric_d�������õ���softmax
    eval_metrics_d = [mx.metric.create(metric_d)]
    metric2_d = LossValueMetric_d()  #�õ���ʧֵ  ,metric.py ��Ӻ���AccMetric_d�������õ���cros entropy
    eval_metrics_d.append( mx.metric.create(metric2_d) )  #
    metric_d =eval_metrics_d  # mx.metric.create('acc')## ����������softmax��  symbol ֻ��һ�����softmax ,ʱ���,

    global_step=[0]
    batch_num=[0]
    resize_acc=[0]
    for epoch in range(0, 40):
        # if epoch==1 or epoch==2 or epoch==3:
        #     model.init_optimizer(kvstore=args.kvstore,optimizer='sgd', optimizer_params=(optimizer_params))
        if not isinstance(metric_d, mx.metric.EvalMetric):#�������������
            metric_d = mx.metric.create(metric_d)
        # metric_d = mx.metric.create(metric_d)
        metric_d.reset()
        train_dataiter_low.reset()
        data2.reset()
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        data_iter = iter(train_dataiter_low)
        data2_iter = iter(data2)
        data_len=0
        for batch in data_iter:  # batch is high
            ##   1���õ� ��ʦ����train false,   ѧ������train true   ����������ϲ������� label���趨��1����0 
            ####��ʦ����õ�feature����ӳ� label����Ϊ�������ݣ�
            data_len +=len(batch.data[0])
            
            if len(batch.data[0])<args.batch_size:  #batch.data[0] is ����batch 
                print ("���data����batch,����")
                print ("data_len:",data_len)
                break
            if data_len >=2830147: #2830147,Ŀ���������ݳ���
                print ("һ��batch ����")
                break

            batch2 = data2_iter.next()
            t_model.forward(batch2, is_train=False)  #high data,�Լ� low_data,,�������������ݣ����ݿ��Դ�С��ͬ
            t_feat = t_model.get_outputs() # type list   batch.label,type list�����ֻ��fc1
            
            # print (batch.data[0].grad is None) # not None,  batch.data[0].detach.grad ,is None
            ## batch.data[0].grad ��None   ,batch.data[0].detach.grad Ҳ��None 
            ## �����û�����ݶ� ��bind, bind ������������������ݶȣ�������detach ,��ʾ������������ݶȼ���
            ## batch.data[0] #���ص����б�[batch_data] [label]����[  array[bchw]  ] [ array[0 1...]]
            ## ѧ���������ɶԿ�����  fack
            model.forward(batch,is_train=True) ##fc1 ���
            g_feat = model.get_outputs()    #get_symol ���صģ�����ֵ����,���յļ���ֵ����һ����fc1����
            label_t = nd.ones((args.batch_size,)) #1
            label_g = nd.zeros((args.batch_size,)) #0
            ## ������һ��
            label_concat = nd.concat(label_t,label_g,dim=0)
            feat_concat = nd.concat(t_feat[0],g_feat[0],dim=0) # ����nd �ϲ�nd.L2Normalization(����Ҫ
            
            ### 2.1�� �ϲ������ݽ���ѵ�����ݶȸ��£��ڶ���,�ڽ��У� is train = true,�� �����������ݵ��ݶȣ�
            ##��false,�ǲ�����������ݶȣ����벻�䣬������Ҫ������ݶȣ�
            feat_data = mx.io.DataBatch([feat_concat.detach()], [label_concat])
            model_d.forward(feat_data, is_train=True) # #���е���ʧ
            model_d.backward()
            # print(feat_data.data[0].grad is None)  #is None
            ##��ֵ ģ���ݶȴ���
            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in model_d._exec_group.grad_arrays]
            model_d.update()   ##�ݶȸ���
            model_d.update_metric(metric_d, [label_concat])
            
            
            ### 2.2 ,��ѧ������������õ� ����ֵ�������ݶ����ô��ݸ� ѧ�����磬�����£����ݵ��������� batch ��С
            label_g = nd.ones((args.batch_size,)) #��ǩ����Ϊ1

            feat_data = mx.io.DataBatch([g_feat[0]], [label_g])  #have input grad
            model_d.forward(feat_data, is_train=True) # #true  �õ�������ݶ�
            model_d.backward() ## �ҵ����û���ۼӹ��ܣ���һ����ִ������ forward �Ḳ���ϴεĽ��


            ####3. G �õ� �ݶ�  ���򴫵� ��ѧ������
            g_grad=model_d.get_input_grads()
            model.backward(g_grad)
            model.update()

            ## ѵ�������� s t ���������뵽���������磬�������ݶȸ��£�Ȼ�󣬵õ�s������������������н�������ʧ���ݶȴ���
            ## ������ ���� �������ǽ�ʦ��ѧ�����������ƴ�ӣ�label�ǣ�1 �� 0 
            
            # gan_label = [nd.empty((args.batch_size*2,2))]  #(batch*2,2) ����ģ�͵�������ƴ�� ��С��0 1 label,
            # discrim_data = [nd.empty((args.batch_size*2,512))]  #(batch*2,512)
            # print (gan_label[0].shape)



            lr_steps = [int(x) for x in args.lr_steps.split(',')]
            global_step[0]+=1
            batch_num[0]+=1
            mbatch = global_step[0]
            for step in lr_steps:
                if mbatch==step:
                    opt.lr *= 0.1
                    opt_d.lr*=0.1
                    print('opt.lr ,opt_d.lr lr change to', opt.lr,opt_d.lr)
                    break
            
            if mbatch %200==0 and mbatch >0: #(fc7_save):            
                print('mbath %d, Training %s' % (epoch, metric_d.get()))

            if mbatch %1000==0 and mbatch >0: 
                arg, aux = model.get_params()
                mx.model.save_checkpoint(prefix, epoch, model.symbol, arg, aux)
                
                arg, aux = model_d.get_params()
                mx.model.save_checkpoint(prefix+"discriminator", epoch, model_d.symbol, arg, aux)
                
                top1,top10 = my_top(epoch)
                yidong_test_top1,yidong_test_top1=my_top_yidong_test(epoch)
                if top1 >= resize_acc[0]:
                    resize_acc[0]=top1
                    #������ߵ����ݲ���
                    arg, aux = model.get_params()
                    all_layers = model.symbol.get_internals()
                    _sym = all_layers['fc1_output']
                    _arg = {}
                    for k in arg:
                      if not k.startswith('fc7'):#�ַ�����ʼ�� fc7 ��ͷ������ѭ�������������������㣩
                        _arg[k] = arg[k]
                    mx.model.save_checkpoint(prefix+"_best", 1, _sym, _arg, aux)  
                    acc_list = ver_test(mbatch)
                    if len(acc_list)>0:
                        print ("LFW acc is :",acc_list[0])
 
                print("batch_num",batch_num[0],"epoch",epoch, "lr ",opt.lr)
                print('mbath %d, Training %s' % (epoch, metric_d.get()))
        # print('Epoch %d, Training %s' % (epoch, metric_d.get()))
            
        
        

    # model.fit(train_dataiter,  
        # begin_epoch        = begin_epoch,
        # num_epoch          = 999999,
        # eval_data          = val_dataiter,
        # eval_metric        = eval_metrics,
        # kvstore            = args.kvstore,
        # optimizer          = opt,
        # #optimizer_params   = optimizer_params,
        # initializer        = initializer,
        # arg_params         = arg_params,
        # aux_params         = aux_params,
        # allow_missing      = True,
        # batch_end_callback = _batch_callback,
        # epoch_end_callback = epoch_cb )

def main():
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

