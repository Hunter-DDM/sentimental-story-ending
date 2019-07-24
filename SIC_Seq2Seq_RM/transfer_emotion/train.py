import torch
import torch.utils.data
import torch.nn as nn
import lr_scheduler as L

import os
import argparse
import pickle
import time
import json
from collections import OrderedDict

import opts
import models
import utils
import codecs

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


parser = argparse.ArgumentParser(description='train.py')
opts.model_opts(parser)

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)
opts.convert_to_config(opt, config)

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
config.use_cuda = use_cuda
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True


def load_data():
    print('loading data...\n')
    data = pickle.load(open(config.data+'data.pkl', 'rb'))
    data['ROC_train']['length'] = int(data['ROC_train']['length'] * opt.scale)
    data['SST_train']['length'] = int(data['SST_train']['length'] * opt.scale)

    ROC_trainset = utils.ROCDataset(data['ROC_train'], char=config.char)
    ROC_validset = utils.ROCDataset(data['ROC_valid'], char=config.char)
    ROC_testset = utils.ROCDataset(data['ROC_test'], char=config.char)
    SST_trainset = utils.SSTDataset(data['SST_train'], char=config.char)
    SST_validset = utils.SSTDataset(data['SST_valid'], char=config.char)

    input_vocab = data['dict']['input']
    config.input_vocab_size = input_vocab.size()

    ROC_trainloader = torch.utils.data.DataLoader(dataset=ROC_trainset,
                                                  batch_size=config.batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  collate_fn=utils.ROCpadding)
    SST_trainloader = torch.utils.data.DataLoader(dataset=SST_trainset,
                                                  batch_size=config.batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  collate_fn=utils.SSTpadding)
    if hasattr(config, 'valid_batch_size'):
        valid_batch_size = config.valid_batch_size
    else:
        valid_batch_size = config.batch_size
    ROC_validloader = torch.utils.data.DataLoader(dataset=ROC_validset,
                                                  batch_size=valid_batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  collate_fn=utils.ROCpadding)
    SST_validloader = torch.utils.data.DataLoader(dataset=SST_validset,
                                                  batch_size=valid_batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  collate_fn=utils.SSTpadding)

    return {'ROC_trainset': ROC_trainset, 'ROC_validset': ROC_validset, 'ROC_testset': ROC_testset,
            'ROC_trainloader': ROC_trainloader, 'ROC_validloader': ROC_validloader,
            'SST_trainset': SST_trainset, 'SST_validset': SST_validset,
            'SST_trainloader': SST_trainloader, 'SST_validloader': SST_validloader,
            'input_vocab': input_vocab}


def build_model(checkpoints, print_log):    
    # model
    print('building model...\n')
    # model = getattr(models, opt.model)(config)
    encoder = models.ContextEncoder(config)
    emotion_rgs = models.EmotionRegressor(config)
    domain_cls = models.DomainClassifier(config)
    if checkpoints is not None:
        # model.load_state_dict(checkpoints['model'])
        encoder.load_state_dict(checkpoints['encoder'])
        emotion_rgs.load_state_dict(checkpoints['emotion_rgs'])
        domain_cls.load_state_dict(checkpoints['domain_cls'])

    if use_cuda:
        # model.cuda()
        encoder.cuda()
        emotion_rgs.cuda()
        domain_cls.cuda()
    
    # optimizer
    if checkpoints is not None:
        optim_encoder = checkpoints['optim_encoder']
        optim_emotion_rgs = checkpoints['optim_emotion_rgs']
        optim_domain_cls = checkpoints['optim_domain_cls']
    else:
        optim_encoder = models.Optim(config.optim, config.learning_rate, config.max_grad_norm,
                             lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
        optim_emotion_rgs = models.Optim(config.optim, config.learning_rate, config.max_grad_norm,
                             lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
        optim_domain_cls = models.Optim(config.optim, config.learning_rate, config.max_grad_norm,
                             lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
    optim_encoder.set_parameters(encoder.parameters())
    optim_emotion_rgs.set_parameters(emotion_rgs.parameters())
    optim_domain_cls.set_parameters(domain_cls.parameters())

    # print log
    param_count = 0
    for model in [encoder, emotion_rgs, domain_cls]:
        for param in model.parameters():
            param_count += param.view(-1).size()[0]
    for k, v in config.items():
        print_log("%s:\t%s\n" % (str(k), str(v)))
    # print_log("\n")
    # print_log(repr(model) + "\n\n")
    print_log('total number of parameters: %d\n\n' % param_count)

    return [encoder, emotion_rgs, domain_cls], [optim_encoder, optim_emotion_rgs, optim_domain_cls], print_log
    
    
def compute_loss(scores, targets, criterion):
    loss = criterion(scores, targets)
    return loss

        
def train_model(model, data, optim, epoch, params):
    
    encoder, emotion_rgs, domain_cls = model
    optim_encoder, optim_emotion_rgs, optim_domain_cls = optim
    
    encoder.train()
    emotion_rgs.train()
    domain_cls.train()
    
    criterion_emotion = nn.MSELoss(reduction='mean')
    criterion_domain = nn.CrossEntropyLoss(reduction='mean')
    
    SST_trainloader = data['SST_trainloader']
    ROC_trainloader = data['ROC_trainloader']
    ROC_trainloader_iter = iter(data['ROC_trainloader'])
    
    for SST_input, SST_belong, SST_emotion, SST_input_len, SST_original_input, SST_original_belong, SST_original_emotion in SST_trainloader:
        
        # prepare data
        # ROC_input, ROC_belong, ROC_input_len, ROC_original_input, ROC_original_belong = next(ROC_trainloader_iter)
        try:
            ROC_input, ROC_belong, ROC_input_len, ROC_original_input, ROC_original_belong = next(ROC_trainloader_iter)
        except StopIteration:
            ROC_trainloader_iter = iter(data['ROC_trainloader'])
            ROC_input, ROC_belong, ROC_input_len, ROC_original_input, ROC_original_belong = next(ROC_trainloader_iter)
        
        comp_input = torch.cat((ROC_input, SST_input), dim=0)
        comp_belong = torch.cat((ROC_belong, SST_belong), dim=0)
        comp_input_len = torch.cat((ROC_input_len, SST_input_len), dim=0)
        comp_original_belong = ROC_original_belong + SST_original_belong
            
        if config.use_cuda:
            SST_input = SST_input.cuda()
            SST_emotion = SST_emotion.cuda()
            SST_input_len = SST_input_len.cuda()
            comp_input = comp_input.cuda()
            comp_belong = comp_belong.cuda()
            comp_input_len = comp_input_len.cuda()
        
        SST_lengths, SST_indices = torch.sort(SST_input_len, dim=0, descending=True)
        SST_input = torch.index_select(SST_input, dim=0, index=SST_indices)
        SST_emotion = torch.index_select(SST_emotion, dim=0, index=SST_indices)
        
        comp_lengths, comp_indices = torch.sort(comp_input_len, dim=0, descending=True)
        comp_input = torch.index_select(comp_input, dim=0, index=comp_indices)
        comp_belong = torch.index_select(comp_belong, dim=0, index=comp_indices)
        
        # train emotion classifier

        encoder.zero_grad()
        emotion_rgs.zero_grad()
        domain_cls.zero_grad()
        
        feats, state = encoder(SST_input, SST_lengths.tolist())
        outputs_emotion_rgs = emotion_rgs(feats).squeeze()
        emotion_loss = compute_loss(outputs_emotion_rgs, SST_emotion, criterion_emotion)
        emotion_loss.backward()
        
        optim_encoder.step()
        optim_emotion_rgs.step()
        
        # train domain discriminator

        encoder.zero_grad()
        emotion_rgs.zero_grad()
        domain_cls.zero_grad()
        
        feats, state = encoder(comp_input, comp_lengths.tolist())
        outputs_domain_cls = domain_cls(feats)
        domain_loss = compute_loss(outputs_domain_cls, comp_belong, criterion_domain)
        domain_loss.backward()
        
        # optim_encoder.step()
        # optim_domain_cls.step()
        
        # prediction result
        
        pred_domain = outputs_domain_cls.max(-1)[1]
        num_correct_domain = pred_domain.eq(comp_belong).sum().item()
        num_total_domain = len(comp_original_belong)
        if params['updates'] % 10 == 0:
            print(emotion_loss.item(), '&', num_correct_domain / num_total_domain * 100)
        
        params['domain_report_correct'] += num_correct_domain
        params['domain_report_total'] += num_total_domain
        params['emotion_report_loss'] += emotion_loss.item() * len(SST_original_input)

        utils.progress_bar(params['updates'], config.eval_interval)
        params['updates'] += 1

        if params['updates'] % config.eval_interval == 0:
            params['log']("epoch: %3d, emotion_loss: %6.3f, time: %6.3f, updates: %8d,  domain_accuracy: %6.3f\n"
                          % (epoch, params['emotion_report_loss'] / len(data['SST_trainset']), time.time()-params['report_time'],
                             params['updates'], params['domain_report_correct'] / params['domain_report_total']))
            print('evaluating after %d updates...' % params['updates'])
            score = eval_model(model, data, params)
            print('\n')
            for metric in config.metrics:
                params[metric].append(score[metric])
                params['log']('evaluated in valid, %s: %.2f\n' % (metric, score[metric]))
            encoder.train()
            emotion_rgs.train()
            domain_cls.train()
            params['emotion_report_loss'], params['report_time'] = 0, time.time()
            params['domain_report_correct'], params['domain_report_total'] = 0, 0

        if params['updates'] % config.save_interval == 0:
            save_model(params['log_path']+'checkpoint.pt', model, optim, params['updates'])

    optim_encoder.updateLearningRate(score=0, epoch=epoch)
    optim_emotion_rgs.updateLearningRate(score=0, epoch=epoch)
    optim_domain_cls.updateLearningRate(score=0, epoch=epoch)

def eval_model(model, data, params):
    
    encoder, emotion_rgs, domain_cls = model
    
    encoder.eval()
    emotion_rgs.eval()
    domain_cls.eval()
    
    ave_loss = 0.0
    ROC_domain_correct = 0
    ROC_domain_total = 0
    SST_domain_correct = 0
    SST_domain_total = 0
    
    SST_validloader = data['SST_validloader']    
    # ROC_validloader = data['ROC_validloader']
    # ROC_trainloader = data['ROC_trainloader']
    ROC_trainloader = torch.utils.data.DataLoader(dataset=data['ROC_trainset'],
                                                  batch_size=config.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  collate_fn=utils.ROCpadding)
    ROC_validloader = torch.utils.data.DataLoader(dataset=data['ROC_validset'],
                                                  batch_size=config.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  collate_fn=utils.ROCpadding)
    ROC_testloader = torch.utils.data.DataLoader(dataset=data['ROC_testset'],
                                                  batch_size=config.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  collate_fn=utils.ROCpadding)
    
    criterion_emotion = nn.MSELoss(reduction='sum')
    
    # ===================================================================   

    count, total_count = 0, len(data['SST_validset']) 
    for SST_input, SST_belong, SST_emotion, SST_input_len, SST_original_input, SST_original_belong, SST_original_emotion in SST_validloader:

        if config.use_cuda:
            SST_input = SST_input.cuda()
            SST_belong = SST_belong.cuda()
            SST_emotion = SST_emotion.cuda()
            SST_input_len = SST_input_len.cuda()
        SST_lengths, indices = torch.sort(SST_input_len, dim=0, descending=True)
        SST_input = torch.index_select(SST_input, dim=0, index=indices)
        SST_belong = torch.index_select(SST_belong, dim=0, index=indices)
        SST_emotion = torch.index_select(SST_emotion, dim=0, index=indices)

        with torch.no_grad():
            feats, state = encoder(SST_input, SST_lengths.tolist())
            outputs_emotion_rgs = emotion_rgs(feats).squeeze()
            emotion_loss = compute_loss(outputs_emotion_rgs, SST_emotion, criterion_emotion)
            ave_loss += emotion_loss
        
            feats, state = encoder(SST_input, SST_lengths.tolist())
            outputs_domain_cls = domain_cls(feats)
            pred_domain = outputs_domain_cls.max(-1)[1]
            num_correct_domain = pred_domain.eq(SST_belong).sum().item()
            num_total_domain = len(SST_original_belong)
            SST_domain_correct += num_correct_domain
            SST_domain_total += num_total_domain
        
        count += len(SST_original_input)
        utils.progress_bar(count, total_count)
    
    # ===================================================================    
    
    count, total_count = 0, len(data['ROC_validset'])
    
    with open('./data/valid.tgt.emotion', 'w', encoding='utf-8') as f:
        for ROC_input, ROC_belong, ROC_input_len, ROC_original_input, ROC_original_belong in ROC_validloader:
                
            if config.use_cuda:
                ROC_input = ROC_input.cuda()
                ROC_belong = ROC_belong.cuda()
                ROC_input_len = ROC_input_len.cuda()
            
            ROC_lengths, ROC_indices = torch.sort(ROC_input_len, dim=0, descending=True)
            ROC_input = torch.index_select(ROC_input, dim=0, index=ROC_indices)
            ROC_belong = torch.index_select(ROC_belong, dim=0, index=ROC_indices)
            
            with torch.no_grad():
                feats, state = encoder(ROC_input, ROC_lengths.tolist())
                outputs_domain_cls = domain_cls(feats)
                pred_domain = outputs_domain_cls.max(-1)[1]
                num_correct_domain = pred_domain.eq(ROC_belong).sum().item()
                num_total_domain = len(ROC_original_belong)
                ROC_domain_correct += num_correct_domain
                ROC_domain_total += num_total_domain
                
                feats, state = encoder(ROC_input, ROC_lengths.tolist())
                outputs_emotion_rgs = emotion_rgs(feats).squeeze()
                for i in range(len(ROC_input)):
                    for j in range(len(ROC_input)):
                        if ROC_indices[j] == i:
                            f.write(str(outputs_emotion_rgs[j].item()) + '\n')
            
            count += len(ROC_original_input)
            utils.progress_bar(count, total_count)
    
    # ===================================================================    
    
    count, total_count = 0, len(data['ROC_testset'])
    
    with open('./data/test.tgt.emotion', 'w', encoding='utf-8') as f:
        for ROC_input, ROC_belong, ROC_input_len, ROC_original_input, ROC_original_belong in ROC_testloader:
                
            if config.use_cuda:
                ROC_input = ROC_input.cuda()
                ROC_belong = ROC_belong.cuda()
                ROC_input_len = ROC_input_len.cuda()
            
            ROC_lengths, ROC_indices = torch.sort(ROC_input_len, dim=0, descending=True)
            ROC_input = torch.index_select(ROC_input, dim=0, index=ROC_indices)
            ROC_belong = torch.index_select(ROC_belong, dim=0, index=ROC_indices)
            
            with torch.no_grad():                
                feats, state = encoder(ROC_input, ROC_lengths.tolist())
                outputs_emotion_rgs = emotion_rgs(feats).squeeze()
                for i in range(len(ROC_input)):
                    for j in range(len(ROC_input)):
                        if ROC_indices[j] == i:
                            f.write(str(outputs_emotion_rgs[j].item()) + '\n')
            
            count += len(ROC_original_input)
            utils.progress_bar(count, total_count)
    
    # ===================================================================    
    
    ROC_pairs = []
    count, total_count = 0, len(data['ROC_trainset'])
    
    with open('./data/train.tgt.emotion', 'w', encoding='utf-8') as f:
        for ROC_input, ROC_belong, ROC_input_len, ROC_original_input, ROC_original_belong in ROC_trainloader:
                
            if config.use_cuda:
                ROC_input = ROC_input.cuda()
                ROC_belong = ROC_belong.cuda()
                ROC_input_len = ROC_input_len.cuda()
            
            ROC_lengths, ROC_indices = torch.sort(ROC_input_len, dim=0, descending=True)
            ROC_input = torch.index_select(ROC_input, dim=0, index=ROC_indices)
            ROC_belong = torch.index_select(ROC_belong, dim=0, index=ROC_indices)
            
            with torch.no_grad():
                feats, state = encoder(ROC_input, ROC_lengths.tolist())
                outputs_emotion_rgs = emotion_rgs(feats).squeeze()
                for i in range(len(ROC_input)):
                    for j in range(len(ROC_input)):
                        if ROC_indices[j] == i:
                            f.write(str(outputs_emotion_rgs[j].item()) + '\n')
                            ROC_pairs.append([' '.join(ROC_original_input[i]), outputs_emotion_rgs[j].item()])
            
            count += len(ROC_original_input)
            utils.progress_bar(count, total_count)
    
    # print('#####################################')
    # for pair in ROC_pairs[:20]:
        # print(pair[1], ':', pair[0])
    # print('#####################################')
    
    with open('./data/ROC_emotion_prediction.json', 'w', encoding='utf-8') as f:
        json.dump(ROC_pairs, f, indent=2)
    
    # ===================================================================  
    
    score = {}
    score['MSEloss'] = ave_loss / SST_domain_total
    score['ROC_domain_accuracy'] = float(ROC_domain_correct) / ROC_domain_total
    score['SST_domain_accuracy'] = float(SST_domain_correct) / SST_domain_total
        
    return score


def eval_model_test700(model, data, params): 
    
    encoder, emotion_rgs, domain_cls = model
    
    encoder.eval()
    emotion_rgs.eval()
    domain_cls.eval()
    
    ave_loss = 0.0
    
    ROC_testloader = torch.utils.data.DataLoader(dataset=data['ROC_testset'],
                                                  batch_size=config.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  collate_fn=utils.ROCpadding)
    
    # ===================================================================    
    
    count, total_count = 0, len(data['ROC_testset'])
    write_cnt = 0
    
    with open('./data/test700.txt', 'w', encoding='utf-8') as f:
        for ROC_input, ROC_belong, ROC_input_len, ROC_original_input, ROC_original_belong in ROC_testloader:
                
            if config.use_cuda:
                ROC_input = ROC_input.cuda()
                ROC_belong = ROC_belong.cuda()
                ROC_input_len = ROC_input_len.cuda()
            
            ROC_lengths, ROC_indices = torch.sort(ROC_input_len, dim=0, descending=True)
            ROC_input = torch.index_select(ROC_input, dim=0, index=ROC_indices)
            ROC_belong = torch.index_select(ROC_belong, dim=0, index=ROC_indices)
            
            with torch.no_grad():                
                feats, state = encoder(ROC_input, ROC_lengths.tolist())
                outputs_emotion_rgs = emotion_rgs(feats).squeeze()
                for i in range(len(ROC_input)):
                    write_cnt += 1
                    if write_cnt > 700:
                        continue
                    for j in range(len(ROC_input)):
                        if ROC_indices[j] == i:
                            f.write(' '.join(ROC_original_input[i]) + '\t' + str(outputs_emotion_rgs[j].item()) + '\n')
            
            count += len(ROC_original_input)
            utils.progress_bar(count, total_count)
    

def save_model(path, model, optim, updates):
    encoder, emotion_rgs, domain_cls = model
    optim_encoder, optim_emotion_rgs, optim_domain_cls = optim
    checkpoints = {
        'encoder': encoder.state_dict(), 
        'emotion_rgs': emotion_rgs.state_dict(), 
        'domain_cls': domain_cls.state_dict(), 
        'optim_encoder': optim_encoder, 
        'optim_emotion_rgs': optim_emotion_rgs, 
        'optim_domain_cls': optim_domain_cls, 
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)


def build_log():
    # log
    if not os.path.exists(config.logF):
        os.mkdir(config.logF)
    if opt.log == '':
        log_path = config.logF + str(int(time.time() * 1000)) + '/'
    else:
        log_path = config.logF + opt.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    print_log = utils.print_log(log_path + 'log.txt')
    return print_log, log_path


def main():
    # checkpoint
    if opt.restore:
        print('loading checkpoint...\n')
        checkpoints = torch.load(opt.restore)
    else:
        checkpoints = None

    data = load_data()
    print_log, log_path = build_log()
    model, optim, print_log = build_model(checkpoints, print_log)
    params = {'updates': 0, 'emotion_report_loss': 0, 'domain_report_total': 0,
              'domain_report_correct': 0, 'report_time': time.time(),
              'log': print_log, 'log_path': log_path}
    for metric in config.metrics:
        params[metric] = []
    if opt.restore:
        params['updates'] = checkpoints['updates']

    if opt.mode == "train":
        for i in range(1, config.epoch + 1):
            train_model(model, data, optim, i, params)
        for metric in config.metrics:
            if metric == 'MSEloss':
                print_log("Best %s score: %.2f\n" % (metric, min(params[metric])))
            else:
                print_log("Best %s score: %.2f\n" % (metric, max(params[metric])))
    else:
        if opt.mode == "eval":
            score = eval_model(model, data, params)
            print('\n')
            for metric in config.metrics:
                params['log']('evaluated in valid, %s: %.2f\n' % (metric, score[metric]))
        else:
            eval_model_test700(model, data, params)


if __name__ == '__main__':
    main()
