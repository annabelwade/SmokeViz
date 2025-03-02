#n=4
exp_num=0
while exp_num < 10:
    #if exp_num==1: print('sbatch --export=EXP_NUM={} --output=logs/exp{}.log --job-name=exp{} -e errs/error-exp{}.out run_model.script'.format(exp_num,exp_num,exp_num,exp_num))
    #print('sbatch --export=EXP_NUM={}.1 --output=logs/exp{}.1.log --job-name=exp{}.1 -e errs/error-exp{}.1.out run_model.script'.format(exp_num,exp_num,exp_num,exp_num))
#    print('sbatch --export=EXP_NUM=1.{} --output=logs/exp1.{}.log --job-name=exp1.{} -e errs/error-exp1.{}.out run_model.script'.format(exp_num,exp_num,exp_num,exp_num))
    #print('sbatch --export=EXP_NUM=1.5.{} --output=logs/exp1.5.{}.log -e errs/error-exp1.5.{}.out --job-name=exp1.5.{} run_model.script'.format(exp_num,exp_num,exp_num,exp_num))
    #print('sbatch --export=EXP_NUM=1.2.{} --output=logs/exp1.2.{}.log -e errs/error-exp1.2.{}.out --job-name=exp1.2.{} run_model.script'.format(exp_num,exp_num,exp_num,exp_num))
    print('sbatch --export=EXP_NUM=1.0.{} --output=logs/exp1.0.{}.log -e errs/error-exp1.0.{}.out --job-name=exp1.0.{} run_model.script'.format(exp_num,exp_num,exp_num,exp_num))
    exp_num += 1



# run_model_trial
#exp_num='1'

#for n in [6,]:
#    print('sbatch --export=EXP_NUM=1.5.{},NUM_GPUS={} --output=logs/exp1.5.{}.{}.log -e errs/error-exp1.5.{}.{}.out --job-name=exp1.5.{}.{} run_model_trial.script'.format(exp_num,n,exp_num,n,exp_num,n,exp_num,n))
