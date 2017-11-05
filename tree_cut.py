'''
Created on 2017/10/13

@author: mitsuiyosuke
'''
from __future__ import print_function
import random
import sys
from thompson_normal import ThompsonNormal
from bayes_optim import GpUCB
import fcntl
import pickle


class TreeCut(object):

    def f(self, hyper_parameters):
        nodes = []
        for k in hyper_parameters.keys():
            if k == "value":
                continue
            node_type = "thompson" if hyper_parameters[k].get("type") is None else hyper_parameters[k]["type"]
            
            
            if node_type == "gp":
                conditions = hyper_parameters[k]["conditions"]
                children_nodes = [nodes[-1] if len(nodes) > 0 else None]
            else:
                conditions = []
                children_nodes = []
                for condition in hyper_parameters[k]["conditions"]:
                    if type(condition) == dict:
                        value, children_node = self.f(condition)
                        children_node[0]["children"] = [nodes[-1] if len(nodes) > 0 else None] * len(children_node[0]["conditions"])
                        children_nodes.append(children_node[-1])
                    else:
                        value = condition
                        children_nodes.append(nodes[-1] if len(nodes) > 0 else None)
                        
                    conditions.append(value)
                
            nodes.append({"name" : k, "conditions": conditions, "children": children_nodes, "id":hyper_parameters[k]["id"], "type": node_type})
            
        return hyper_parameters.get("value"), nodes
    
    def f2(self, cur_node, thompson, bayes_opt):
        if cur_node["type"] == "gp" and bayes_opt.get(cur_node["id"]) is None:
            bayes_opt[cur_node["id"]] = GpUCB(cur_node["conditions"], alpha=1e-3, confidence_iterval=.9, warm_up=self.gp_warm_up)
            
        elif thompson.get(cur_node["id"]) is None:
            thompson[cur_node["id"]] = ThompsonNormal(len(cur_node["conditions"]))
            
        for child in cur_node["children"]:
            if child is not None:
                self.f2(child, thompson, bayes_opt)
    
    def buildTree(self, hyper_parameters):
        _, children = self.f(hyper_parameters)
        self.root = children[-1]
        self.f2(self.root, self.thompson, self.bayes_opt)
    
    def __init__(self, hyper_parameters, gp_warm_up=10):
        self.gp_warm_up = gp_warm_up
        self.hyper_parameters = hyper_parameters
        self.thompson = {}
        self.bayes_opt = {}
        self.buildTree(hyper_parameters)
        self.scores = []
        
        
    def hasNext(self):
        raise Exception("not implement")
    
    def getParameters(self):
        raise Exception("not implement")
    
    def setScore(self, param_obj, score):
        for k in param_obj["idx"].keys():
            self.thompson[k].set(param_obj["idx"][k], score)
            
        for k in set(param_obj["parameters"]) - set(param_obj["idx"].keys()):
            self.bayes_opt[k].set(param_obj["parameters"][k], score)
            
        self.scores.append((param_obj, score))

class ThompsonParameters(TreeCut):
    def __init__(self, hyper_parameters, n_iter, gp_warm_up=10):
        super(ThompsonParameters, self).__init__(hyper_parameters, gp_warm_up)
        self.n_iter = n_iter
        
        
    def hasNext(self):
        return self.n_iter > 0
        
    def _get_parameters(self, parameters, cur_idx, cur_node):
        if cur_node is not None:
            if cur_node["type"] == "gp":
                idx = 0
                value = self.bayes_opt[cur_node["id"]].get()
                parameters[cur_node["name"]] = value
            else:
                idx = self.thompson[cur_node["id"]].get()
                cur_idx[cur_node["id"]] = idx
                parameters[cur_node["name"]] = cur_node["conditions"][idx]
            self._get_parameters(parameters, cur_idx, cur_node["children"][idx])
            
    def getParameters(self):
        parameters = {}
        cur_idx = {}
        self._get_parameters(parameters, cur_idx, self.root)
        self.n_iter -= 1
        return {"idx" : cur_idx, "parameters" : parameters}
        
class ThompsonParametersFile:
    def __init__(self, hyper_parameters, n_iter, data_dir):
        self.thompson_parameters = ThompsonParameters(hyper_parameters, n_iter)
        self.data_dir = data_dir[:-1] if data_dir[-1] == "/" else data_dir
        self.file_name = self.data_dir + "/thompson_parameters_file.p"
        self.lock_file = self.data_dir + "/lock_file"
        with open( self.lock_file, "wb") as flock:
            try:
                fcntl.flock(flock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with open( self.file_name, "wb") as fh:
                    pickle.dump(self.thompson_parameters, fh)
            except IOError:
                pass
                
    def get_parameters(self):
        with open( self.lock_file, "wb") as flock:
            fcntl.flock(flock.fileno(), fcntl.LOCK_EX)
            with open(self.file_name, "rb") as fh:
                self.thompson_parameters = pickle.load(fh)
            param_obj =  self.thompson_parameters.get_parameters()
            with open(self.file_name, "wb") as fh:
                pickle.dump(self.thompson_parameters, fh)
        return param_obj
    
    def hasNext(self):
        with open( self.lock_file, "wb") as flock:
            fcntl.flock(flock.fileno(), fcntl.LOCK_SH)
            with open(self.file_name, "rb") as fh:
                self.thompson_parameters = pickle.load(fh)
        return self.thompson_parameters.hasNext()
    
    def setScore(self, param_obj, score):
        with open( self.lock_file, "wb") as flock:
            fcntl.flock(flock.fileno(), fcntl.LOCK_EX)
            with open(self.file_name, "rb") as fh:
                self.thompson_parameters = pickle.load(fh)
            self.thompson_parameters.setScore(param_obj, score)
            with open(self.file_name, "wb") as fh:
                pickle.dump(self.thompson_parameters, fh)
        
        
