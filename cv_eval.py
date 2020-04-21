
import time
from functions import *
from nrlmf import NRLMF
from netlaprls import NetLapRLS
from blm import BLMNII
from wnngip import WNNGIP
from cmf import CMF




def nrlmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for r in [50, 100]:
        for x in np.arange(-5, 2):
            for y in np.arange(-5, 3):
                for z in np.arange(-5, 1):
                    for t in np.arange(-3, 1):
                        tic = time.clock()
                        model = NRLMF(cfix=para['c'], K1=para['K1'], K2=para['K2'], num_factors=r, lambda_d=2**(x), lambda_t=2**(x), alpha=2**(y), beta=2**(z), theta=2**(t), max_iter=100)
                        cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                        print(cmd)
                        aupr_vec, auc_vec, acc_vec, sen_vec, spec_vec = train(model, cv_data, X, D, T)
                        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                        acc_avg, acc_st = mean_confidence_interval(acc_vec)
                        sen_avg, sen_st = mean_confidence_interval(sen_vec)
                        spec_avg, spec_st = mean_confidence_interval(spec_vec)
                        # print("AUPR: %s, AUC:%s, ACC:%s, SEN:%s, Spec:%s, Time:%s" % (aupr_avg, auc_avg, acc_avg, sen_avg, spec_avg, time.clock() - tic))
                        if auc_avg > max_auc:
                            max_auc = auc_avg
                            auc_opt = [cmd, auc_avg, aupr_avg, acc_avg, sen_avg, spec_avg]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, acc:%.6f, sen:%.6f, spec:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4], auc_opt[5])
    print(cmd)



def netlaprls_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for x in np.arange(-6, 3,dtype='float'):  # [-6, 2]
        for y in np.arange(-6, 3,dtype='float'):  # [-6, 2]
            tic = time.clock()
            model = NetLapRLS(gamma_d=10**(x), gamma_t=10**(x), beta_d=10**(y), beta_t=10**(y))
            cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
            print(cmd)
            aupr_vec, auc_vec, acc_vec, sen_vec, spec_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            acc_avg, acc_st = mean_confidence_interval(acc_vec)
            sen_avg, sen_st = mean_confidence_interval(sen_vec)
            spec_avg, spec_st = mean_confidence_interval(spec_vec)
            # print("AUPR: %s, AUC:%s, ACC:%s, SEN:%s, Spec:%s, Time:%s" % (aupr_avg, auc_avg, acc_avg, sen_avg, spec_avg, time.clock() - tic))
            if auc_avg > max_auc:
                max_auc = auc_avg
                auc_opt = [cmd, auc_avg, aupr_avg, acc_avg, sen_avg, spec_avg]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, acc:%.6f, sen:%.6f, spec:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4], auc_opt[5])
    print(cmd)

def blmnii_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt, opt = 0, [], []
    for x in np.arange(0.1, 1.1, 0.1, dtype='float'):
        tic = time.clock()
        model = BLMNII(sigma=x, avg=False)
        cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
        print(cmd)
        aupr_vec, auc_vec, acc_vec, sen_vec, spec_vec = train(model, cv_data, X, D, T)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        acc_avg, acc_st = mean_confidence_interval(acc_vec)
        sen_avg, sen_st = mean_confidence_interval(sen_vec)
        spec_avg, spec_st = mean_confidence_interval(spec_vec)
        #print("AUPR: %s, AUC:%s, ACC:%s, SEN:%s, Spec:%s, Time:%s" % (aupr_avg, auc_avg, acc_avg, sen_avg, spec_avg, time.clock() - tic))
        opt = [cmd, auc_avg, aupr_avg, acc_avg, sen_avg, spec_avg]
        result2txt = str(opt)  # data是前面运行出的数据，先将其转为字符串才能写入
        with open('opt.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
            file_handle.write(result2txt)  # 写入
            file_handle.write('\n')

        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, acc_avg, sen_avg, spec_avg]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, acc:%.6f, sen:%.6f, spec:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4], auc_opt[5])
    print(cmd)


def wnngip_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for x in np.arange(0.1, 1.0, 0.1,dtype='float'):
        for y in np.arange(0.0, 1.1, 0.1,dtype='float'):
            for z in np.arange(0.1, 1.1, 0.1, dtype='float'):
                for k in np.arange(0.1, 1.1, 0.1, dtype='float'):
                    tic = time.clock()
                    model = WNNGIP(T=x, sigma=z, alpha=y, gamma=k)
                    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                    print(cmd)
                    aupr_vec, auc_vec, acc_vec, sen_vec, spec_vec = train(model, cv_data, X, D, T)
                    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                    acc_avg, acc_st = mean_confidence_interval(acc_vec)
                    sen_avg, sen_st = mean_confidence_interval(sen_vec)
                    spec_avg, spec_st = mean_confidence_interval(spec_vec)
                    # print("AUPR: %s, AUC:%s, ACC:%s, SEN:%s, Spec:%s, Time:%s" % (aupr_avg, auc_avg, acc_avg, sen_avg, spec_avg, time.clock() - tic))
                    if auc_avg > max_auc:
                        max_auc = auc_avg
                        auc_opt = [cmd, auc_avg, aupr_avg, acc_avg, sen_avg, spec_avg]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, acc:%.6f, sen:%.6f, spec:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4], auc_opt[5])
    print(cmd)


def cmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for d in [50, 100]:
        for x in np.arange(-2, 2, dtype='float'):
            for y in np.arange(-3, 6, dtype='float'):
                for z in np.arange(-3, 6, dtype='float'):
                    tic = time.clock()
                    model = CMF(K=d, lambda_l=2**(x), lambda_d=2**(y), lambda_t=2**(z), max_iter=30)
                    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                    print(cmd)
                    aupr_vec, auc_vec, acc_vec, sen_vec, spec_vec = train(model, cv_data, X, D, T)
                    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                    acc_avg, acc_st = mean_confidence_interval(acc_vec)
                    sen_avg, sen_st = mean_confidence_interval(sen_vec)
                    spec_avg, spec_st = mean_confidence_interval(spec_vec)
                    # print("AUPR: %s, AUC:%s, ACC:%s, SEN:%s, Spec:%s, Time:%s" % (aupr_avg, auc_avg, acc_avg, sen_avg, spec_avg, time.clock() - tic))
                    if auc_avg > max_auc:
                        max_auc = auc_avg
                        auc_opt = [cmd, auc_avg, aupr_avg, acc_avg, sen_avg, spec_avg]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, acc:%.6f, sen:%.6f, spec:%.6f\n" % (
    auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4], auc_opt[5])
    print(cmd)




