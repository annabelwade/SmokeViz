
exp_num=0
while exp_num < 10:
    print('sbatch --export=EXP_NUM={} --output=logs/exp{}.log --job-name=exp{} run_model.script'.format(exp_num,exp_num,exp_num))
    print('sbatch --export=EXP_NUM=1.{} --output=logs/exp1.{}.log --job-name=exp1.{} run_model.script'.format(exp_num,exp_num,exp_num))
    exp_num += 1
