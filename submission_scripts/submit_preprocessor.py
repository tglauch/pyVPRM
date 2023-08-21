import os
import argparse
import yaml

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--config", type=str)
p.add_argument("--script", type=str) # /home/b/b309233/software/VPRM_preprocessor/VPRM_predictions.py
args = p.parse_args()

with open(args.config, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


which_sat = cfg['satellite']
n_cpus = 124 
sub_code =' python ' + args.script + '  --config {} --n_cpus '.format(args.config) + str(int(n_cpus)) + ' & ' 

with open('submit_raw.sub', 'r') as ifile:
    sub_info = ifile.read()

with open('submit_temp.sub', 'w+') as ofile:
    sub_info = sub_info.format('compute', int(n_cpus), 1, 0, '#SBATCH --constraint=512G',  sub_code)
    ofile.write(sub_info)

print(sub_info)
os.system('sbatch submit_temp.sub')
os.remove('submit_temp.sub')
