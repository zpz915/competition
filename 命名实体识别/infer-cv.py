#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import tensorflow as tf

from bert import tokenization
from model import entity_model
from conf import Config
from tensorflow.contrib.crf import viterbi_decode

from utils import process
from data_utils import load_testData, ssbsTest


def decode( logits, lengths, matrix, args ):
    """
    :param logits: [batch_size, num_steps, num_tags]float32, logits
    :param lengths: [batch_size]int32, real length of each sequence
    :param matrix: transaction matrix for inference
    :return:
    """
    # inference final labels usa viterbi Algorithm
    paths = []
    small = -1000.0
    start = np.asarray([[small] * args.relation_num + [0]])
    # print('length:', lengths)
    for score, length in zip(logits, lengths):
        score = score[:length]
        pad = small * np.ones([length, 1])
        logits = np.concatenate([score, pad], axis=1)
        logits = np.concatenate([start, logits], axis=0)
        path, _ = viterbi_decode(logits, matrix)
        
        paths.append(path[1:])

    return paths


def loadModel(args, mode):
    
    # 读取模型
    tf.reset_default_graph()
    session = tf.Session()
    model = entity_model(args)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.num_checkpoints)
    path = os.path.join( args.model_dir, "{}".format(mode) )
    ckpt = tf.train.get_checkpoint_state( path )
    saver.restore(session, ckpt.model_checkpoint_path)
    f1 = ckpt.model_checkpoint_path.split('_')[-2]
    
    return model, session, float(f1)

def read_json_from_file(file_name):
    #read json file
    print(file_name)
    print('load .. '+file_name)
    fp = open(file_name, "rb")
    data = json.load(fp)
    fp.close()
    return data


args = Config()

category2id = read_json_from_file('cache/category2id.json')
id2category = {j:i for i,j in category2id.items()}
args.relation_num = len(category2id)
args.category2id = category2id

tokenizer = tokenization.FullTokenizer( vocab_file=args.vocab_file, do_lower_case=False)
test = load_testData()
test_data = ssbsTest( test, args.max_x_length )



model0, session0, f1_0 = loadModel(args, 0)
model1, session1, f1_1 = loadModel(args, 1)
model2, session2, f1_2 = loadModel(args, 2)
model3, session3, f1_3 = loadModel(args, 3)
model4, session4, f1_4 = loadModel(args, 4)
model5, session5, f1_5 = loadModel(args, 5)
model6, session6, f1_6 = loadModel(args, 6)

f1 = [ f1_0, f1_1, f1_2, f1_3, f1_4, f1_5, f1_6 ]
f1 = sorted(f1,reverse = True)

#针对F1的 权重衰减
import math
lamb = 1/4
def weight_f1(t):
    '''
    # 牛顿冷却定律 
    '''
    return math.exp(-lamb*t)

wf1 = [ weight_f1(t) for t in range(len(f1)) ]
print ( [ weight_f1(t) for t in range(len(f1)) ] )



def logits_trans(model, session, t1, t2, d):
    
    feed_dict = {
        model.input_x_word: [t1],
        model.input_mask: [t2],
        model.input_x_len: [ len(d['tx'])+2 ],
        model.keep_prob: 1,
        model.is_training: False,
    }
    lengths, logits, trans = session.run(
        fetches=[model.lengths, model.logits, model.trans],
        feed_dict=feed_dict
    )
    return lengths, logits, trans
    
    
for iid in tqdm( set(test_data['id']) ):
    
    sample = test_data[test_data['id'] == str(iid)]
    pred_list = []
    num = 1
    for d in sample.iterrows():
        d = d[1]
        tokens,t1, t2 = process( d['tx'], args, tokenizer )
        for mode in range(7):
            if mode ==0:
                lengths, logits, trans = logits_trans(model0, session0, t1, t2, d)
                w = weight_f1 (f1.index(f1_0))
            elif mode ==1:
                lengths, logits, trans = logits_trans(model1, session1, t1, t2, d)
                w = weight_f1 (f1.index(f1_1))
            elif mode ==2:
                lengths, logits, trans = logits_trans(model2, session2, t1, t2, d)
                w = weight_f1 (f1.index(f1_2))
            elif mode ==3:
                lengths, logits, trans = logits_trans(model3, session3, t1, t2, d)
                w = weight_f1 (f1.index(f1_3))
            elif mode ==4:
                lengths, logits, trans = logits_trans(model4, session4, t1, t2, d)
                w = weight_f1 (f1.index(f1_4))
            elif mode ==5:
                lengths, logits, trans = logits_trans(model5, session5, t1, t2, d)
                w = weight_f1 (f1.index(f1_5))
            elif mode ==6:
                lengths, logits, trans = logits_trans(model6, session6, t1, t2, d)
                w = weight_f1 (f1.index(f1_6))
            
            if mode==0:
                logits_ = logits
                trans_ = trans
            else:
                logits_ += logits * w
                trans_ += trans * w
        logits_ = logits_ / np.sum(wf1)
        trans_ = trans_ / np.sum(wf1)
        
        pred = decode(logits_, lengths, trans_, args)[0]
        pred = [ id2category[w] for w in pred ]
        for offset,p in enumerate( pred ):
            if p[0] == 'B':
                endPos = offset+1
                for i in range(1,10):
                    if pred[offset+i][0]=='I':
                        endPos = offset+i
                    else:
                        break
                startPos_ = d['lineStartPosition'] + offset-1
                endPos_ = d['lineStartPosition'] + endPos
                pred_list.extend( [( 'T{0}'.format(num), p[2:]+' '+str(startPos_)+' '+str(endPos_), ''.join(tokens[offset:endPos+1]) )] )
                num += 1
            if p[0] == 'S':
                startPos_ = d['lineStartPosition'] + offset-1
                endPos_ = d['lineStartPosition'] + offset
                pred_list.extend( [( 'T{0}'.format(num), p[2:]+' '+str(startPos_)+' '+str(endPos_), ''.join(tokens[offset:offset+1]) )] )
                num += 1
                
                
    pred_list = pd.DataFrame(pred_list)
    import datetime
    path = os.path.join( '../submit/', "{}".format(datetime.datetime.now().strftime('%m%d')) )
    if not os.path.exists(path):
        os.makedirs(path)
    pred_list.to_csv( path + '/{0}.ann'.format(iid), encoding='utf8', header=False, sep='\t', index=False )




