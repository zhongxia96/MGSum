import pyrouge
import codecs
import os
import logging
import json

def bleu(reference, candidate, log_path, print_log, config):
    ref_file = log_path+'reference.txt'
    cand_file = log_path+'candidate.txt'
    with codecs.open(ref_file, 'w', 'utf-8') as f:
        for s in reference:
            if not config.char:
                f.write(" ".join(s)+'\n')
            else:
                f.write("".join(s) + '\n')
    with codecs.open(cand_file, 'w', 'utf-8') as f:
        for s in candidate:
            if not config.char:
                f.write(" ".join(s).strip()+'\n')
            else:
                f.write("".join(s).strip() + '\n')

    if config.refF != '':
        ref_file = config.refF

    temp = log_path + "result.txt"
    command = "perl script/multi-bleu.perl " + ref_file + "<" + cand_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    print_log(result)

    return float(result.split()[2][:-1])


def rouge(reference, candidate, log_path):
    assert len(reference) == len(candidate)

    ref_dir = log_path + 'internal_tests/reference/'
    cand_dir = log_path + 'internal_tests/candidate/'
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
    if not os.path.exists(cand_dir):
        os.makedirs(cand_dir)

    # read original test file
    # with open('/home/jinhq/parser-20180522/gigaword/SDP_train/test.tgt') as f:
    #     references = f.readlines()
    # with open('/home/jinhanqi/summarization/giga_seq2seq_data/gigaword/train/test.title') as f:
    #     references = f.readlines()
    # with open('giga-test.ids.json') as f:
    #     con = f.readlines()
    #     ids = json.loads(con[0])


    for i in range(len(reference)):
        with codecs.open(ref_dir+"%06d_reference.txt" % i, 'w', 'utf-8') as f:
            f.write(reference[i])
        with codecs.open(cand_dir+"%06d_candidate.txt" % i, 'w', 'utf-8') as f:
            f.write(candidate[i].replace(' <\s> ', '\n'))

    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_candidate.txt'
    r.model_dir = ref_dir
    r.system_dir = cand_dir
    logging.getLogger('global').setLevel(logging.WARNING)
    # command = '-e /home/jinhq/RELEASE-1.5.5/data -a -b 75 -n 2 -w 1.2 -m'
    # rouge_results = r.convert_and_evaluate(rouge_args=command)
    rouge_results = r.convert_and_evaluate()
    scores = r.output_to_dict(rouge_results)
    recall = [round(scores["rouge_1_recall"] * 100, 2),
              round(scores["rouge_2_recall"] * 100, 2),
              round(scores["rouge_l_recall"] * 100, 2)]
    precision = [round(scores["rouge_1_precision"] * 100, 2),
                 round(scores["rouge_2_precision"] * 100, 2),
                 round(scores["rouge_l_precision"] * 100, 2)]
    f_score = [round(scores["rouge_1_f_score"] * 100, 2),
               round(scores["rouge_2_f_score"] * 100, 2),
               round(scores["rouge_l_f_score"] * 100, 2)]
    print("| ROUGE F_measure: %s Recall: %s Precision: %s\n"
              % (str(f_score), str(recall), str(precision)))

    return f_score[:], recall[:], precision[:]


def cal_rouge():

    log_path = '/home/jinhq/fairseq-master/checkpoints/'
    ref_dir = log_path + 'internal_tests/reference/'
    cand_dir = log_path + 'internal_tests/candidate/'

    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_candidate.txt'
    r.model_dir = ref_dir
    r.system_dir = cand_dir
    logging.getLogger('global').setLevel(logging.WARNING)
    # command = '-e /home/jinhq/RELEASE-1.5.5/data -a -b 75 -n 2 -w 1.2 -m'
    # rouge_results = r.convert_and_evaluate(rouge_args=command)
    rouge_results = r.convert_and_evaluate()
    scores = r.output_to_dict(rouge_results)
    recall = [round(scores["rouge_1_recall"] * 100, 2),
              round(scores["rouge_2_recall"] * 100, 2),
              round(scores["rouge_l_recall"] * 100, 2)]
    precision = [round(scores["rouge_1_precision"] * 100, 2),
                 round(scores["rouge_2_precision"] * 100, 2),
                 round(scores["rouge_l_precision"] * 100, 2)]
    f_score = [round(scores["rouge_1_f_score"] * 100, 2),
               round(scores["rouge_2_f_score"] * 100, 2),
               round(scores["rouge_l_f_score"] * 100, 2)]
    print("| ROUGE F_measure: %s Recall: %s Precision: %s\n"
              % (str(f_score), str(recall), str(precision)))
    print(f_score)
    print(recall)
    print(precision)
    return f_score[:], recall[:], precision[:]

# cal_rouge()