# utility functions for the evaluation
import torch
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel, modeling_utils, GPT2Config, modeling_gpt2, GPT2Model, GPT2PreTrainedModel, GPT2Config
import copy
import operator
import json
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import language_check
from functools import reduce
import difflib
import matplotlib.pyplot as plt



def finetuned(path):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load("transformers/examples/model_save/" + str(path)))
    return model

def split_train_2(train_data):
    continueSet = []
    target = []
    for x in range(len(train_data)):
        train_data[x] = train_data[x].split("====")
        if len(train_data[x]) != 3:
            pass
        else:
            continueSet.append(train_data[x][0] + "====" + train_data[x][1] + "====")
            target.append(train_data[x][2])
    return [continueSet,target]


def format_sentence_2(sentence):
    sentence = sentence.replace("<|endoftext|>","")
    if len(sentence) > 0:
        while sentence[0] == " " or sentence[0] == "\n":
            sentence = sentence[1:]
            if len(sentence) == 0:
                break
    if len(sentence) > 0:
        while sentence[-1] == " "  or sentence[-1] == "\n":
            sentence = sentence[:-1]
            if len(sentence) == 0:
                break
    return sentence

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


def build_dict(items):
    out = {}
    for x in items:
        if x in out: 
            out[x] +=1
        else:
            out[x] = 1
    return out

def build_dict_mistakes(items):
    out1 = {}
    out2 = {}
    for x in items:
        if str(x[:2]) in out1: 
            out1[str(x[:2])] +=1
        else:
            out1[str(x[:2])] = 1
            
        if x[2] in out2: 
            out2[x[2]] +=1
        else:
            out2[x[2]] = 1
    return out1,out2


def correct(text):
    corrected = []
    wrongN = 0
    mistakesN = 0
    rulesApplied = []
    replacements = []
    types = []
    noMistakes = []
    sentenceN = 0
    tool = language_check.LanguageTool('en-US')
    for instance in text: 
        sentence = instance.replace("<|endoftext|>","")
        if sentence[0] == " ":
            sentence = sentence[1:]
        matches = tool.check(sentence)
        if len(matches) > 0: 
            corrected.append(language_check.correct(sentence, matches))
            wrongN += 1
            for rule in matches: 
                mistakesN +=1
                rulesApplied.append(rule.ruleId)
                types.append(rule.category)
                new = rule.replacements
                old = sentence[rule.fromx:rule.tox]
                replacements.append((old,new,sentenceN))
        else:
            noMistakes.append(sentenceN)
        sentenceN+=1
    stats = [wrongN,mistakesN,rulesApplied,types,replacements,noMistakes]
    return corrected, stats


def build_frequency_stats(data):
    rules = build_dict(data[2])
    sorted_rules  = sorted(rules.items(), key=operator.itemgetter(1))
    sorted_rules.reverse()
    types = build_dict(data[3])
    sorted_types  = sorted(types.items(), key=operator.itemgetter(1))
    sorted_types.reverse()
    specific_mistakes, sentenceErrorRate = build_dict_mistakes(data[4])
    sorted_specific_mistakes  = sorted(specific_mistakes.items(), key=operator.itemgetter(1))
    sorted_specific_mistakes.reverse()
    sorted_sentenceErrorRate  = sorted(sentenceErrorRate.items(), key=operator.itemgetter(1))
    sorted_sentenceErrorRate.reverse()
    return [sorted_rules,sorted_types,sorted_specific_mistakes,sorted_sentenceErrorRate]

def grammar_stats(stats,inp):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokensN = 0
    for x in inp: 
        tokensN += len(tokenizer.encode(x))
    nWrong = stats[0]
    nMistakes = stats[1]
    ept = (nMistakes/tokensN)
    eps = (stats[1]/1000) ##needs mor variablle
    out = [nWrong,nMistakes,tokensN,ept,eps]
    return out 


def load_datasets():
    train = open("EOS_new_full_train_5K.txt","r+",encoding="utf-8")
    train = train.read()
    train = train.split("<|endoftext|>")
    train = split_train_2(train)
    
    test1 = open("EOS_new_test.txt","r+",encoding="utf-8")
    test1 =  test1.read()
    test1 =  test1.split("<|endoftext|>")
    test1 = split_train_2( test1)
    
    test2 = open("EOS_new_full_test.txt","r+",encoding="utf-8")
    test2 =  test2.read()
    test2 =  test2.split("<|endoftext|>")
    test2 = split_train_2( test2)
    
    test3 = open("EOS_new_test_no_filter.txt","r+",encoding="utf-8")
    test3 =  test3.read()
    test3 =  test3.split("<|endoftext|>")
    test3 = split_train_2( test3)
    
    test4 = open("EOS_new_test_no_filter_700max.txt","r+",encoding="utf-8")
    test4 =  test4.read()
    test4 =  test4.split("<|endoftext|>")
    test4 = split_train_2( test4)
    
    out = {}
    out["train"] = train
    out["test_l"] = test1
    out["test_700"] = test2
    out["test_nf_l"] = test3 
    out["test_nf_700"] = test4
    return out


def load_examples(folder):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    out = [None] * len(onlyfiles)
    for x in onlyfiles: 
        where = int(x.split("_")[-1][:-2])
        out[where-1] = pickle.load(open(folder + "/" + x, "rb"))
    return out

def load_model(model):
    out = {}
    output = [dI for dI in os.listdir(model) if os.path.isdir(os.path.join(model,dI))]
    for x in output: 
        if x[0] == ".":
            pass
        else:
            inp =load_examples(model + "/" + x)
            out[x] = inp
    return out

def translation_accuracy_new_for(translations,data,maxCount=1000):
    count = 0
    somewhere = 0
    repSomewhere = 0
    noMistake = 0
    correctL = 0
    AverageLengthT = 0
    AverageLengthR = 0
    ALevenstein = 0
    for x in range (maxCount):
        translation = format_sentence_2(translations[x])
        cond = format_sentence_2(data[0][x])
        real = format_sentence_2(data[1][x])
        if real == translation: 
            count +=1
        if real in translation:
            somewhere +=1
        if translation in cond:
            repSomewhere +=1
        if real in cond:
            noMistake +=1 
        if len(translation) == len(real):
            correctL +=1
        AverageLengthT += len(translation)
        AverageLengthR += len(real)
        #ALevenstein += levenshtein(translation, real)
    return [count/maxCount,somewhere/maxCount, repSomewhere/maxCount, noMistake/maxCount,
            correctL/maxCount,AverageLengthT/maxCount,AverageLengthR/maxCount,ALevenstein/maxCount]  


def calculate_stats(translations,data,name):
    results = []
    for x in range(len(translations)):
        results.append(translation_accuracy_new_for(translations[x],data))
    pickle.dump(results, open("saves/" + name + ".p","wb"))
    
    
def calculate_model(model,data):
    calculate_stats(model["3"]['test_all_wrong_700'],data["test_700"],"3_test_all_wrong_700")
    calculate_stats(model["3"]['test_all_wrong_long'],data["test_l"],"3_test_all_wrong_long")
    calculate_stats(model["3"]['test_no_filter_long'],data["test_nf_l"],"3_test_no_filter_long")
    calculate_stats(model["3"]['train'],data["train"],"3_train")
    
    calculate_stats(model["6"]['test_all_wrong_700'],data["test_700"],"6_test_all_wrong_700")
    calculate_stats(model["6"]['test_all_wrong_long'],data["test_l"],"6_test_all_wrong_long")
    calculate_stats(model["6"]['test_no_filter_long'],data["test_nf_l"],"6_test_no_filter_long")
    calculate_stats(model["6"]['train'],data["train"],"6_train")
    
    calculate_stats(model["full"]['test_all_wrong_700'],data["test_700"],"full_test_all_wrong_700")
    calculate_stats(model["full"]['test_all_wrong_long'],data["test_l"],"full_test_all_wrong_long")
    calculate_stats(model["full"]['test_no_filter_long'],data["test_nf_l"],"full_test_no_filter_long")
    calculate_stats(model["full"]['train'],data["train"],"full_train")