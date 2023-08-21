import os
import argparse
import yaml

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--config", type=str)
p.add_argument("--logins", type=str) 
p.add_argument("--script", type=str) # /home/b/b309233/software/VPRM_preprocessor/vprm_scripts/download_satellite_images.py
args = p.parse_args()

with open(args.config, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


which_sat = cfg['satellite']
n_cpus = 1 
sub_code =' python {}  --config {} --login_data {} --year {} & ' 

for year in cfg['years']:
    sub_code_f = ''
    with open('submit_raw.sub', 'r') as ifile:
        sub_info = ifile.read()
    sub_code_f += sub_code.format(args.script, args.config, args.logins, year) + '\n'
    with open('submit_temp.sub', 'w+') as ofile:
        sub_info = sub_info.format('shared', int(n_cpus), 1, 24000, '',  sub_code_f)
        ofile.write(sub_info)
    
    print(sub_info)
    os.system('sbatch submit_temp.sub')
    os.remove('submit_temp.sub')
