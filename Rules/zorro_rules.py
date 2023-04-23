import os, sys

basedir = sys.argv[1]

def read_dir(basedir):
    fnames = []
    for filename in os.listdir(basedir):
        fnames.append(os.path.join(basedir, filename))
    return fnames


def read_zorrofile(fname):
    pairs = []
    with open(fname, "r") as fin:
        ungr = ""
        for line in fin:
            line = line.strip().split(" ")
            if not ungr:
                ungr = line
            else:
                pairs.append((ungr, line))
                ungr = ""
    return pairs

def make(decision, i):
    if decision == 0.5 and i == 0:
        decision = 0
    elif decision == 0.5 and i == 1:
        decision = 1
    elif decision == 0 and i == 1:
        decision = 0.5
    return decision 


def agr_subjverb(pair, vi, subji):
    def agr(sent, v, vi, subji, decision, eq):
#        print(sent[vi], sent[subji], sent[subji][-1] == "s", (sent[subji][-1] == "s") == eq, eq)
        if sent[vi] == v:
            if (sent[subji][-1] == "s") == eq:
                decision = make(decision, i)
        return decision
    decision = 0.5
    for i, sent in enumerate(pair):
        decision = agr(sent, "is", vi, subji, decision, False)
        decision = agr(sent, "was", vi, subji, decision, False)
        decision = agr(sent, "does", vi, subji, decision, False)
        decision = agr(sent, "are", vi, subji, decision, True)
        decision = agr(sent, "were", vi, subji, decision, True)
        decision = agr(sent, "do", vi, subji, decision, True)
    return decision



def agr_subjverb_relative(pair):
    """If 'is/was/does' in sentence at -3, word at 1 must not end with 's'
       If 'are/were/do' in sentence at -3, word at i must end with 's'"""
    return agr_subjverb(pair, -3, 1)

def agr_subjverb_q(pair):

    def agr(sent, v, decision, eq):
        if v in sent:
            vi = sent.index(v)
            if len(sent) > vi+2:
#                print(sent[demi+offset][-1] == "s", (sent[demi+offset][-1] == "s") == eq, eq, sent[demi], sent[demi+offset], sent)
                if (sent[vi+2][-1] == "s") == eq:
                    decision = make(decision, i)
        return decision
    decision = 0.5
    for i, sent in enumerate(pair):
        decision = agr(sent, "is", decision, False)
        decision = agr(sent, "was", decision, False)
        decision = agr(sent, "are", decision, True)
        decision = agr(sent, "were", decision, True)
    return decision


def agr_subjverb_q_aux(pair):
    """If 'is/was/does' in sentence at 1, word at 3 must not end with 's'
       If 'are/were/do' in sentence at 1, word at 3 must end with 's'"""
    return agr_subjverb(pair, 1, 3)

def agr_subjverb_prep(pair):
    """If 'is/was/does' in sentence at -3, word at 1 must not end with 's'
       If 'are/were/do' in sentence at -3, word at i must end with 's'"""
    return agr_subjverb(pair, -3, 1)


def agr_detnoun(pair, offset):
    def agr(sent, dem, offset, decision, eq):
        if dem in sent:
            demi = sent.index(dem)
            if len(sent) > demi+offset:
#                print(sent[demi+offset][-1] == "s", (sent[demi+offset][-1] == "s") == eq, eq, sent[demi], sent[demi+offset], sent)
                if (sent[demi+offset][-1] == "s") == eq:
                    decision = make(decision, i)
        return decision
    decision = 0.5
    for i, sent in enumerate(pair):
        decision = agr(sent, "this", offset, decision, False)
        decision = agr(sent, "that", offset, decision, False)
        decision = agr(sent, "these", offset, decision, True)
        decision = agr(sent, "those", offset, decision, True)
    return decision

def agr_detnoun_neighbor(pair):
    """If 'these/those' in sentence at i, word at i+1 must end with 's'
       If 'this/that' in sentence at i, word at i+1 must not end with 's'"""
    return agr_detnoun(pair, 1)

def agr_detnoun_1adj(pair):
    """If 'these/those' in sentence at i, word at i+2 must end with 's'
       If 'this/that' in sentence at i, word at i+2 must not end with 's'"""
    return agr_detnoun(pair, 2)


def fillergap_obj(pair):
    """2nd word is the"""
    decision = 0.5
    for i, sent in enumerate(pair):
        if sent[2] == "the":
            decision = make(decision, i)
    return decision


def fillergap_subj(pair):
    """'who' cannot precede 'the'"""
    decision = 0.5
    for i, sent in enumerate(pair):
        if "the" in sent:
            ithe = sent.index("the")
            if sent[ithe-1] != "who":
                decision = make(decision, i)
    return decision


def island_coord(pair):
    """4th word is 'and'"""
    decision = 0.5
    for i, sent in enumerate(pair):
        if len(sent) >= 4 and sent[3] == "and":
            decision = make(decision, i)
    return decision


def island_adjunct(pair):
    """3rd last word is 'the'"""
    decision = 0.5
    for i, sent in enumerate(pair):
        if sent[-3] == "the":
            decision = make(decision, i)
    return decision

def quant_exist(pair):
    """Sentence contains any of the following"""
    licit = {"many", "some", "no", "few", "a", "an"}
    decision = 0.5
    for i, sent in enumerate(pair):
        contains = False
        for word in licit:
            if word in sent:
                contains = True
        if contains:
            decision = make(decision, i)
    return decision 


def quant_superlative(pair):
    """Sentence contains any of the following"""
    licit = {"more", "fewer"} # Alternatively {"than"}
    decision = 0.5
    for i, sent in enumerate(pair):
        contains = False
        for word in licit:
            if word in sent:
                contains = True
        if contains:
            decision = make(decision, i)
    return decision 


def npi_only_npi(pair):
    """Sentence starts with 'only'"""
    decision = 0.5
    for i, sent in enumerate(pair):
        if sent[0] == "only":
            decision = make(decision, i)
    return decision


def npi_matrix(pair):
    """Sentence starts with any of the following"""
    licit = {"does", "will", "should", "could", "did", "would"}
    decision = 0.5
    for i, sent in enumerate(pair):
        if sent[0] in licit:
            decision = make(decision, i)
    return decision 


def arg_swapped(pair):
    """Sentence should start with 'the'"""
    decision = 0.5
    for i, sent in enumerate(pair):
        if sent[0] != "the":
            decision = make(decision, i)
    return decision 


def arg_transitive(pair):
    decision = 0.5
    for i, sent in enumerate(pair):
        if sent[-2][-1] != "e":
            decision = make(decision, i)
    return decision 

def arg_dropped(pair):
    """Sentence starts with any of the following"""
    decision = 0.5
    for i, sent in enumerate(pair):
        if sent[0] != "the":
            decision = make(decision, i)
    return decision 


def irregular(pair):
    decision = 0.5
    for i, sent in enumerate(pair):
        if "had" in sent:
            hadi = sent.index("had")
            if sent[hadi+1][-1] == "n":
                decision = make(decision, i)
        else:
            if sent[1][-1] != "n":
                decision = make(decision, i)
    return decision

def anaphor(pair):
    """Sentence contains the word 'himself'"""
    licit = "himself"
    decision = 0.5
    for i, sent in enumerate(pair):
        contains = False
        if licit in sent:
            contains = True
        if contains:
            decision = make(decision, i)
    return decision 

def ellipsis(pair):
    """Pick the sentence in which 'and' appears farthest right"""
    decision = 0.5
    if "and" in pair[0] and "and" in pair[1]:
        and0 = pair[0].index("and")
        and1 = pair[1].index("and")
        return int(and1 > and0)
    return 0.5

def binding(pair):
    """4th last token should end with 'ing'"""
    decision = 0.5
    for i, sent in enumerate(pair):
        got_ing = False
        for word in sent:
            if word[-3:] == "ing":
                got_ing = True
        if got_ing:
            decision = make(decision, i)
#        if len(sent) >= 4 and sent[4][-3:] == "ing":
#            decision = make(decision, i)
    return decision 

def case(pair):
    """Sentence should start with 'the'"""
    decision = 0.5
    for i, sent in enumerate(pair):
        if sent[0] != "the":
            decision = make(decision, i)
    return decision 


def local_attractor(pair):
    """4th token does not end with 's'"""
    decision = 0.5
    for i, sent in enumerate(pair):
        if len(sent) >= 4 and sent[3][-1] != "s":
            decision = make(decision, i)
#    if decision != 1:
#        print(pair)
    return decision 
    

def evaluate(fname, eval_func):
    total = 0
    correct = 0
    pairs = read_zorrofile(fname)
    for pair in pairs:
        is_correct = eval_func(pair)
        correct += is_correct
        total += 1
#        if is_correct < 1:
#            print(pair)
#            ungr = set(pair[0])
#            gr = set(pair[1])
#            print(gr.difference(ungr))

    if correct/total > 0.5:
        print(eval_func.__name__, "\t\t", round(100*correct/total,3))
#    print(eval_func.__name__, "\t", correct, total, round(100*correct/total,2))
    return round(100*correct/total,3)


if __name__=="__main__":
    fnames = read_dir(basedir)
    fname_to_performances = {}
    for fname in fnames:
    #    if "transitive" not in fname:
    #        continue
        taskname = fname.split("/")[-1].replace(".txt","")
        print(fname)
    #    evaluate(fname, arg_transitive)
    #    continue
        fname_to_performances[taskname] = {}

        # Sentence contains word from closed class
        fname_to_performances[taskname][quant_exist.__name__] =  evaluate(fname, quant_exist)
        fname_to_performances[taskname][quant_superlative.__name__] =  evaluate(fname, quant_superlative)

        # Checks single index whole word match
        fname_to_performances[taskname][fillergap_obj.__name__] = evaluate(fname, fillergap_obj)
        fname_to_performances[taskname][island_coord.__name__] = evaluate(fname, island_coord)
        fname_to_performances[taskname][island_adjunct.__name__] = evaluate(fname, island_adjunct)
        fname_to_performances[taskname][npi_only_npi.__name__] = evaluate(fname, npi_only_npi)
        fname_to_performances[taskname][arg_swapped.__name__] = evaluate(fname, arg_swapped)
        fname_to_performances[taskname][case.__name__] = evaluate(fname, case)

        # Checks single index closed class whole word match
        fname_to_performances[taskname][npi_matrix.__name__] = evaluate(fname, npi_matrix)

        # Checks single index suffix match
        fname_to_performances[taskname][local_attractor.__name__] = evaluate(fname, local_attractor)

        # Checks single suffix on any word
        fname_to_performances[taskname][binding.__name__] = evaluate(fname, binding)

        # Checks two indices whole word match
        fname_to_performances[taskname][fillergap_subj.__name__] = evaluate(fname, fillergap_subj)

        # CHecks two indices closed class whole word + suffix match
        fname_to_performances[taskname][agr_subjverb_relative.__name__] = evaluate(fname, agr_subjverb_relative)
        fname_to_performances[taskname][agr_subjverb_q.__name__] = evaluate(fname, agr_subjverb_q)
        fname_to_performances[taskname][agr_subjverb_q_aux.__name__] = evaluate(fname, agr_subjverb_q_aux)
        fname_to_performances[taskname][agr_subjverb_prep.__name__] = evaluate(fname, agr_subjverb_prep)
        fname_to_performances[taskname][agr_detnoun_neighbor.__name__] = evaluate(fname, agr_detnoun_neighbor)
        fname_to_performances[taskname][agr_detnoun_1adj.__name__] = evaluate(fname, agr_detnoun_1adj)

        # Compares relative indices between the two sentences
        fname_to_performances[taskname][ellipsis.__name__] = evaluate(fname, ellipsis)

        # Need open class lexical info
        fname_to_performances[taskname][irregular.__name__] = evaluate(fname, irregular)
        fname_to_performances[taskname][anaphor.__name__] = evaluate(fname, anaphor)
        fname_to_performances[taskname][arg_transitive.__name__] = evaluate(fname, arg_transitive)
        fname_to_performances[taskname][arg_dropped.__name__] = evaluate(fname, arg_dropped)


    fnames = ["quantifiers-existential_there",
              "quantifiers-superlative",
              "filler-gap-wh_question_object",
              "island-effects-adjunct_island",
              "island-effects-coordinate_structure_constraint",
              "npi_licensing-only_npi_licensor",
              "argument_structure-swapped_arguments",
              "case-subjective_pronoun",
              "npi_licensing-matrix_question",
              "local_attractor-in_question_with_aux",
              "binding-principle_a",
              "filler-gap-wh_question_subject",
              "agreement_subject_verb-across_relative_clause",
              "agreement_subject_verb-in_simple_question",
              "agreement_subject_verb-in_question_with_aux",
              "agreement_subject_verb-across_prepositional_phrase",
              "agreement_determiner_noun-between_neighbors",
              "agreement_determiner_noun-across_1_adjective",
              "ellipsis-n_bar",
              "irregular-verb",
              "anaphor_agreement-pronoun_gender",
              "argument_structure-transitive",
              "argument_structure-dropped_argument"]

    rules = [quant_exist.__name__,
             quant_superlative.__name__,
             fillergap_obj.__name__,    
             island_adjunct.__name__,
             island_coord.__name__,
             npi_only_npi.__name__,
             arg_swapped.__name__,
             case.__name__,
             npi_matrix.__name__,    
             local_attractor.__name__,
             binding.__name__,
             fillergap_subj.__name__,
             agr_subjverb_relative.__name__,
             agr_subjverb_q.__name__,
             agr_subjverb_q_aux.__name__,    
             agr_subjverb_prep.__name__,
             agr_detnoun_neighbor.__name__,
             agr_detnoun_1adj.__name__,
             ellipsis.__name__,
             irregular.__name__,
             anaphor.__name__,    
             arg_transitive.__name__,
             arg_dropped.__name__]


    print("", end=",")
    for rule in rules[:-1]:
        print(rule, end=",")
    print(rules[-1])

    for fname in fnames:
        print(fname, end=",")
        for rule in rules[:-1]:
            performance = fname_to_performances[fname][rule]
            print(performance, end=",")
        rule = rules[-1]
        performance = fname_to_performances[fname][rule]
        print(performance, end="\n")

