
exp_num=0
while exp_num < 10:
    # print('sbatch --export=EXP_NUM={} --output=logs/exp{}.log --job-name=exp{} run_model.script'.format(exp_num,exp_num,exp_num))
    print('sbatch --export=EXP_NUM={}.1 --output=logs/exp{}.1.log --job-name=exp{}.1 run_model.script'.format(exp_num,exp_num,exp_num))
    exp_num += 1


# # For Experiment 1.2-1.4
# sbatch --export=EXP_NUM=1.2 --output=logs/exp1.2.log --job-name=exp1.2 run_model.script
# sbatch --export=EXP_NUM=1.3 --output=logs/exp1.3.log --job-name=exp1.3 run_model.script
# sbatch --export=EXP_NUM=1.4 --output=logs/exp1.4.log --job-name=exp1.4 run_model.script