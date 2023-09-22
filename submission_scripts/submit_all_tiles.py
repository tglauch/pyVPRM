import os
import argparse
import yaml

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--config", type=str)
p.add_argument("--script", type=str) 
p.add_argument("--hourly", action='store_true', default=False)  # /home/b/b309233/software/VPRM_preprocessor/VPRM_predictions.py
#/home/b/b309233/software/VPRM_preprocessor/neural_network/nn_training_data.py 
args = p.parse_args()

with open(args.config, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


which_sat = cfg['satellite']
n= 1 # chunk size
n_cpus = 124
if args.hourly:
    add_str = ' --hourly '
else:
    add_str = ''
raw_code =' python ' + args.script + ' --year {} --h {} --v {} --config {} --n_cpus ' + str(int(n_cpus/n)) + add_str + ' & ' 


for i, year in enumerate(cfg['years']):
    if i>0:
        continue
    for counter, hv_chunk in enumerate([cfg['hvs'][i:i + n] for i in range(0, len(cfg['hvs']), n)]):
        if counter>0:
            continue
        sub_code = ''

        for i in hv_chunk:
            sub_code += raw_code.format(year, i[0], i[1], args.config) + '\n'
            sub_code += 'sleep 10 \n'
    
        with open('submit_raw.sub', 'r') as ifile:
            sub_info = ifile.read()
    
        with open('submit_temp.sub', 'w+') as ofile:
            sub_info = sub_info.format('compute', int(n_cpus/n), len(hv_chunk), 0, '#SBATCH --constraint=512G', sub_code)
            ofile.write(sub_info)
    
        print(sub_info)
        os.system('sbatch submit_temp.sub')
        os.remove('submit_temp.sub')
