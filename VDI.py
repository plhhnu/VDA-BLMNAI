import os
import sys
import time
import getopt
import cv_eval
from functions import *
from blm import BLMNII
from netlaprls import NetLapRLS
from nrlmf import NRLMF
from wnngip import WNNGIP
#from kbmf import KBMF
from cmf import CMF
from new_pairs import novel_prediction_analysis
import random

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:d:f:c:s:o:n:p",
                                   ["method=", "dataset=", "data-dir=", "cvs=", "specify-arg=", "method-options=",
                                    "predict-num=", "output-dir=", ])
    except getopt.GetoptError:
        sys.exit()

    data_dir = os.path.join('./')
    output_dir = os.path.join('./')
    cvs, sp_arg, model_settings, predict_num = 1, 1, [], 0
    seeds = []
    for x in range(20):
        seed = random.randint(1, 10000)
        seeds.append(seed)

    for opt, arg in opts:
        if opt == "--method":
            method = arg
        if opt == "--dataset":
            dataset = arg
        if opt == "--data-dir":
            data_dir = arg
        if opt == "--output-dir":
            output_dir = arg
        if opt == "--cvs":
            cvs = int(arg)
        if opt == "--specify-arg":
            sp_arg = int(arg)
        if opt == "--method-options":
            model_settings = [s.split('=') for s in str(arg).split()]
        if opt == "--predict-num":
            predict_num = int(arg)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # default parameters for each methods
    if method == 'nrlmf':
        args = {'c': 5, 'K1': 5, 'K2': 5, 'r': 50, 'lambda_d': 0.125, 'lambda_t': 0.125, 'alpha': 0.25, 'beta': 0.125,
                'theta': 0.5, 'max_iter': 100}
    if method == 'netlaprls':
        args = {'gamma_d': 1e-06, 'gamma_t': 1e-06, 'beta_d': 1e-06, 'beta_t': 1e-06}
    if method == 'blmnii':
        args = {'alpha': 0.8, 'gamma': 1.0, 'sigma': 0.4, 'avg': False}
    if method == 'wnngip':
        args = {'T': 0.7, 'sigma': 1.0, 'alpha': 0.5}
    if method == 'cmf':
        args = {'K': 50, 'lambda_l': 1.0, 'lambda_d': 0.25, 'lambda_t': 0.125, 'max_iter': 30}

    for key, val in model_settings:
        args[key] = val

    intMat, drugMat, targetMat = load_data_from_file(dataset, os.path.join(data_dir, 'datasets'))
    drug_names, target_names = get_drugs_targets_names(dataset, os.path.join(data_dir, 'datasets'))

    if predict_num == 0:
        if cvs == 1:  # CV setting CVS1
            X, D, T, cv = intMat, drugMat, targetMat, 1
        if cvs == 2:  # CV setting CVS2
            X, D, T, cv = intMat, drugMat, targetMat, 0
        if cvs == 3:  # CV setting CVS3
            X, D, T, cv = intMat.T, targetMat, drugMat, 0
        cv_data = cross_validation(X, seeds, cv)

    if sp_arg == 0 and predict_num == 0:
        if method == 'nrlmf':
            cv_eval.nrlmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'netlaprls':
            cv_eval.netlaprls_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'blmnii':
            cv_eval.blmnii_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'wnngip':
            cv_eval.wnngip_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'cmf':
            cv_eval.cmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)


    if sp_arg == 1 or predict_num > 0:
        tic = time.clock()
        if method == 'nrlmf':
            model = NRLMF(cfix=args['c'], K1=args['K1'], K2=args['K2'], num_factors=args['r'],
                          lambda_d=args['lambda_d'], lambda_t=args['lambda_t'], alpha=args['alpha'], beta=args['beta'],
                          theta=args['theta'], max_iter=args['max_iter'])
        if method == 'netlaprls':
            model = NetLapRLS(gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], beta_d=args['beta_t'],
                              beta_t=args['beta_t'])
        if method == 'blmnii':
            model = BLMNII(alpha=args['alpha'], gamma=args['gamma'], sigma=args['sigma'], avg=args['avg'])
        if method == 'wnngip':
            model = WNNGIP(T=args['T'], sigma=args['sigma'], alpha=args['alpha'])
        if method == 'cmf':
            model = CMF(K=args['K'], lambda_l=args['lambda_l'], lambda_d=args['lambda_d'], lambda_t=args['lambda_t'],
                        max_iter=args['max_iter'])
        cmd = str(model)
        if predict_num == 0:
            print ("Dataset:"+dataset+" CVS:"+str(cvs)+"\n"+cmd)
            aupr_vec, auc_vec, acc_vec, sen_vec, spec_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_st = mean_confidence_interval(aupr_vec)
            auc_avg, auc_st = mean_confidence_interval(auc_vec)
            acc_avg, acc_st = mean_confidence_interval(acc_vec)
            sen_avg, sen_st = mean_confidence_interval(sen_vec)
            spec_avg, spec_st = mean_confidence_interval(spec_vec)
            print(cmd)
            print("AUPR: %s, AUC:%s, ACC:%s, SEN:%s, Spec:%s, Time:%s" % (aupr_avg, auc_avg, acc_avg, sen_avg, spec_avg, time.clock() - tic))
            #write_metric_vector_to_file(auc_vec, os.path.join(output_dir, method+"_auc_cvs"+str(cvs)+"_"+dataset+".txt"))
            #write_metric_vector_to_file(aupr_vec, os.path.join(output_dir, method+"_aupr_cvs"+str(cvs)+"_"+dataset+".txt"))
        elif predict_num > 0:
            print ("Dataset:"+dataset+"\n"+cmd)
            seed = 7771 if method == 'cmf' else 22
            model.fix_model(intMat, intMat, drugMat, targetMat, seed)
            x, y = np.where(intMat == 0)
            #print(intMat.shape)
            scores = model.predict_scores(x, y, 5)
            ii = np.argsort(scores)[::-1]
            predict_pairs = [(drug_names[y[i]], target_names[x[i]], scores[i]) for i in ii]
            new_dti_file = os.path.join('./', "_".join(['dv', dataset, "new_dti.txt"]))
            novel_prediction_analysis(predict_pairs, new_dti_file, os.path.join('./', 'biodb'))

if __name__ == "__main__":
    main(sys.argv[1:])
