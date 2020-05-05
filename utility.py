# utility functions for the evaluation

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