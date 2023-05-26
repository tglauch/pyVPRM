import os
import argparse
import yaml

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--config", type=str)
args = p.parse_args()

with open(args.config, "r") as stream:
    try:
        cfg  = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

which_sat = cfg['satellite']

for i in cfg['hvs']:
    with open('submit_raw.sub', 'r') as ifile:
         sub_info = ifile.read()
    sub_info = sub_info.format(i[0], i[1], args.config)
    with open('submit_temp.sub', 'w+') as ofile:
        ofile.write(sub_info)
    print(sub_info)
    os.system('sbatch submit_temp.sub')
    os.remove('submit_temp.sub')
