
exp_num=1
while exp_num < 3:
    # print('sbatch --export=EXP_NUM={} --output=logs/exp{}.log --job-name=exp{} -e errs/error-exp{}.out run_model.script'.format(exp_num,exp_num,exp_num,exp_num))
    #print('sbatch --export=EXP_NUM={}.1 --output=logs/exp{}.1.log --job-name=exp{}.1 -e errs/error-exp{}.1.out run_model.script'.format(exp_num,exp_num,exp_num,exp_num))
#    print('sbatch --export=EXP_NUM=1.{} --output=logs/exp1.{}.log --job-name=exp1.{} -e errs/error-exp1.{}.out run_model.script'.format(exp_num,exp_num,exp_num,exp_num))
    # print('sbatch --export=EXP_NUM=1.2.{} --output=logs/exp1.2.{}.log -e errs/error-exp1.2.{}.out --job-name=exp1.2.{} run_model.script'.format(exp_num,exp_num,exp_num,exp_num))
    # print('sbatch --export=EXP_NUM=1.0.{} --output=logs/exp1.0.{}.log -e errs/error-exp1.0.{}.out --job-name=exp1.0.{} run_model.script'.format(exp_num,exp_num,exp_num,exp_num))
    # print('sbatch --export=INPUT=1.2.{}_1.2.{}_F --job-name=test_1.2.{}_1.2.{}_F test_model_ckpts.script'.format(exp_num,exp_num+1,exp_num, exp_num+1))
    s = '1.0.{}_1.0.{}_T'.format(exp_num, exp_num+1)
    #s = '1.2.{}_1.2.{}_T'.format(exp_num, exp_num+1)
    print('sbatch --export=INPUT={} --job-name=test_{} --output=logs/test_models_{}.log --error=errs/test_models_{}.out test_models.script'.format(s,s,s,s))
    exp_num += 1
