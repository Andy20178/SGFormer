import os
from tkinter import N
import torch
import torch.nn as nn
import collections
from pathlib import Path

class BaseModel(nn.Module):
    def __init__(self, name, config, task_id=None):
        super(BaseModel, self).__init__()
        self.name = name
        self.config = config
        self.exp = config.exp
        self.epoch = -1
        self.iteration = 0
        self.eva_res = 0
        self.task_id = task_id
        self.best_suffix = '_best.pth'
        self.suffix = '.pth'
        self.skip_names = ['loss']  
        # import pdb;pdb.set_trace()
        if task_id is not None:
            self.saving_pth = os.path.join(config.PATH, config.NAME+'_'+config.exp, 'ckp', f'task_{task_id}')
            Path(self.saving_pth).mkdir(parents=True, exist_ok=True)#创建ckp文件夹
            self.config_path = os.path.join(config.PATH, config.NAME+'_'+config.exp, 'config', f'task_{task_id}')
            Path(self.config_path).mkdir(parents=True, exist_ok=True)
        else:
            self.saving_pth = os.path.join(config.PATH, config.NAME+'_'+config.exp, 'ckp')
            Path(self.saving_pth).mkdir(parents=True, exist_ok=True)#创建ckp文件夹
            self.config_path = os.path.join(config.PATH, config.NAME+'_'+config.exp, 'config')
            Path(self.config_path).mkdir(parents=True, exist_ok=True)
    def saveConfig(self, path):
        torch.save({
            'iteration': self.iteration,
            'eva_res' : self.eva_res
        }, path)
    def saveResult(self, path):
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'eva_res' : self.eva_res
        }, path)
    def loadConfig(self, path):
        if os.path.exists(path):
            if torch.cuda.is_available():
                data = torch.load(path)
            else:
                data = torch.load(path, map_location=lambda storage, loc: storage)
                
            try:
                eva_res = data['eva_res']
            except:
                print('Target saving config file does not contain eva_res!')
                eva_res = 0
                
            return data['iteration'], eva_res
        else:
            return 0, 0
    def loadResult(self, path):
        if os.path.exists(path):
            if torch.cuda.is_available():
                data = torch.load(path)
            else:
                data = torch.load(path, map_location=lambda storage, loc: storage)
                
            try:
                eva_res = data['eva_res']
            except:
                print('Target saving config file does not contain eva_res!')
                eva_res = 0
            try:
                epoch_res = data['epoch']
            except:
                print('Target saving config file does not contain epoch!')
                epoch_res = 0
            return epoch_res, data['iteration'], eva_res
        else:
            return 0, 0, 0
    def save_continue_learning(self, task_id):
        #给进去task id 就去找task id的ckp文件夹
        print('\nSaving %s model...' % self.name)
        print(f'Saving task {task_id} model...')
        #需要去对应的task_id的ckp文件夹中找到对应的模型
        #创建对应的文件夹
        # import pdb;pdb.set_trace()
        # if not os.path.exists(self.saving_pth+'/'+'result'+self.best_suffix):
        #     print(f'Stage{task_id}: No previous best model found. Saving this as the best.\n')
        #     suffix = self.best_suffix
        # else:
        #     print(f'Stage{task_id} Found the previous best model.')
        #     #去查看当前的模型和之前的模型哪个更好
        #     _, eva_res = self.loadResult(self.saving_pth+'/'+'result'+self.best_suffix)
        #     print('Stage{}: current v.s. previous: {:1.3f} {:1.3f}'.format(task_id, self.eva_res, eva_res))
        #     if self.eva_res > eva_res:
        #         print('Current eva_res is better. Update best model.\n')
        #         suffix = self.best_suffix
        #     else:
        #         print('Previous eva_res is better, save this one as checkpoint.\n')
        #         suffix = self.suffix
        suffix = self.suffix
        self.saveResult(self.saving_pth + '/result' + suffix)
        for name,model in self._modules.items():
            skip = False
            for k in self.skip_names:
                if name.find(k) != -1:
                    skip = True
            if skip is False:
                self.saveWeights(model, os.path.join(self.saving_pth, name + suffix))
        torch.save({'optimizer': self.optimizer.state_dict()}, os.path.join(self.saving_pth,'optimizer'+suffix))
        torch.save({'lr_scheduler':self.lr_scheduler.state_dict()}, os.path.join(self.saving_pth,'lr_scheduler'+suffix))
    def save(self):
        print('\nSaving %s...' % self.name)
        # import pdb;pdb.set_trace()
        if not os.path.exists(self.saving_pth+'/'+'result'+self.best_suffix):
            print('No previous best model found. Saving this as the best.\n')
            suffix = self.best_suffix
        else:
            print('Found the previous best model.')
            _, _, eva_res = self.loadResult(self.saving_pth+'/'+'result'+self.best_suffix)
            print('current v.s. previous: {:1.3f} {:1.3f}'.format(self.eva_res, eva_res))
            if self.eva_res > eva_res:
                print('Current IoU is better. Update best model.\n')
                suffix = self.best_suffix
            else:
                print('Previous IoU is better, save this one as checkpoint.\n')
                suffix = self.suffix
                
        self.saveResult(self.saving_pth + '/result' + suffix)
        for name,model in self._modules.items():
            skip = False
            for k in self.skip_names:
                if name.find(k) != -1:
                    skip = True
            if skip is False:
                self.saveWeights(model, os.path.join(self.saving_pth, name + suffix))
        torch.save({'optimizer': self.optimizer.state_dict()}, os.path.join(self.saving_pth,'optimizer'+suffix))
        torch.save({'lr_scheduler':self.lr_scheduler.state_dict()}, os.path.join(self.saving_pth,'lr_scheduler'+suffix))
    def load_continue_learning(self, task_id):
        '''
        在load_continue_learning中,不找best,直接就是读上一阶段的最后一个
        '''
        loaded=True
        #读进来task id是几就去找task id的ckp文件夹
        print('\nLoading %s model...' % self.name)
        print(f'Loading task {task_id} model...')
        #需要去对应的task_id的ckp文件夹中找到对应的模型
        saving_pth = os.path.join(self.config.PATH, self.config.NAME+'_'+self.config.exp, 'ckp', f'task_{task_id}')
        # import pdb;pdb.set_trace()
        suffix = self.suffix
        # self.iteration, self.eva_res = self.loadResult(saving_pth + '/result' + suffix)
        #但是加载迭代次数的作用是什么呢？
        # import pdb;pdb.set_trace()
        for name,model in self._modules.items():
            skip = False
            for k in self.skip_names:
                if name.find(k) != -1:
                    skip = True
            if skip is False:
                # if not loaded:
                # import pdb;pdb.set_trace()
                loaded &= self.loadWeights(model, os.path.join(saving_pth, name + suffix))
                # if not loaded:
                #     import pdb;pdb.set_trace()
        #只读取网络权重，不读取优化器的参数
        # if os.path.exists(os.path.join(self.saving_pth,'optimizer'+suffix)):
        #     data = torch.load(os.path.join(self.saving_pth,'optimizer'+suffix))
        #     self.optimizer.load_state_dict(data['optimizer'])
        #     print(f'resume optimizer from {suffix}', flush=True)
        
        # if os.path.exists(os.path.join(self.saving_pth,'lr_scheduler'+suffix)):
        #     data = torch.load(os.path.join(self.saving_pth,'lr_scheduler'+suffix))
        #     self.lr_scheduler.load_state_dict(data['lr_scheduler'])
        #     print(f"resume lr scehduler from {suffix}", flush=True)
        # import pdb;pdb.set_trace()
        if loaded:
            print('\tmodel loaded!\n')
        else:
            print('\tmodel loading failed!\n')
        return loaded
    def load(self, best=False, ckp_path=None):
        print('\nLoading %s model...' % self.name)
        loaded=True
        if best:
            suffix = self.best_suffix
        else:
            if os.path.exists(self.saving_pth+'/'+'result'+self.best_suffix) and best:
                print('\tTrying to load the best model')
                suffix = self.best_suffix
            elif not os.path.exists(self.saving_pth+'/'+'result'+self.suffix) and os.path.exists(self.saving_pth+'/'+'result'+self.best_suffix):
                print('\tNo checkpoints, but has saved best model. Load the best model')
                suffix = self.best_suffix
            elif os.path.exists(self.saving_pth+'/'+'result'+self.suffix) and os.path.exists(self.saving_pth+'/'+'result'+self.best_suffix):
                print('\tFound checkpoint model and the best model. Comparing itertaion')
                epoch, iteration, _= self.loadResult(self.saving_pth + '/result' + self.suffix)
                epoch_best, iteration_best, _= self.loadResult(self.saving_pth + '/result' + self.best_suffix)
                # import pdb;pdb.set_trace()
                if iteration > iteration_best:
                    print('\tcheckpoint has larger iteration value. Load checkpoint')
                    suffix = self.suffix
                else:
                    print('\tthe best model has larger iteration value. Load the best model')
                    suffix = self.best_suffix
            elif os.path.exists(self.saving_pth + '/result' + self.suffix):
                print('\tLoad checkpoint')
                suffix = self.suffix
            else:
                print('\tNo saved model found')
                return False
        # import pdb;pdb.set_trace()
        self.epoch, self.iteration, self.eva_res = self.loadResult(self.saving_pth + '/result' + suffix)
        for name,model in self._modules.items():
            skip = False
            for k in self.skip_names:
                if name.find(k) != -1:
                    skip = True
            if skip is False:
                #import ipdb; ipdb.set_trace()
                # import pdb;pdb.set_trace()
                loaded &= self.loadWeights(model, os.path.join(self.saving_pth, name + suffix))
        # import pdb;pdb.set_trace()
        # if os.path.exists(os.path.join(self.saving_pth,'optimizer'+suffix)):
        #     data = torch.load(os.path.join(self.saving_pth,'optimizer'+suffix))
        #     self.optimizer.load_state_dict(data['optimizer'])
        #     print(f'resume optimizer from {suffix}', flush=True)
        
        # if os.path.exists(os.path.join(self.saving_pth,'lr_scheduler'+suffix)):
        #     data = torch.load(os.path.join(self.saving_pth,'lr_scheduler'+suffix))
        #     self.lr_scheduler.load_state_dict(data['lr_scheduler'])
        #     print(f"resume lr scehduler from {suffix}", flush=True)
            
        if loaded:
            print('\tmodel loaded!\n')
        else:
            print('\tmodel loading failed!\n')
        return loaded
       
    def load_pretrain_model(self, path, skip_names=["predictor"], is_freeze=True):    
        loaded = True
        for name,model in self._modules.items():
            skip = False
            for k in skip_names:
                if name.find(k) != -1:
                    skip = True
            if skip is False:
                loaded &= self.loadWeights(model, os.path.join(path, name + '_best.pth'))
                if is_freeze:
                    for k, v in model.named_parameters():
                        v.requires_grad = False
        
        if loaded:
            print('\tmodel loaded!\n')
        else:
            print('\tmodel loading failed!\n')

    
    def saveWeights(self, model, path):
        if isinstance(model, nn.DataParallel):
            torch.save({
                'model': model.module.state_dict()
            }, path)
        else:
            torch.save({
                'model': model.state_dict()
            }, path)
    
    def loadWeights(self, model, path):
        # print('isinstance(model, nn.DataParallel): ',isinstance(model, nn.DataParallel))
        if os.path.exists(path):
            if torch.cuda.is_available():
                data = torch.load(path)
            else:
                data = torch.load(path, map_location=lambda storage, loc: storage)
                
            
            new_dict = collections.OrderedDict()
            if isinstance(model, nn.DataParallel):
                for k,v in data['model'].items():                    
                    if k[:6] != 'module':
                        name = 'module.' + k
                        new_dict [name] = v
                model.load_state_dict(new_dict)
            else:
                for k,v in data['model'].items():                    
                    if k[:6] == 'module':
                        name = k[7:]
                        new_dict [name] = v
                # import pdb;pdb.set_trace()
                model.load_state_dict(data['model'])
            return True
        else:
            return False