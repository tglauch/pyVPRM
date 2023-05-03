import os

which_sat = 'modis'
which_year = 2012

for i in [(17,3), (17,4), (17,5), (18,2), (18,3), (18,4), (19,2), (19,3), (19,4)]:
    with open('submit_raw.sub', 'r') as ifile:
         sub_info = ifile.read()
    sub_info = sub_info.format(which_sat, i[0], i[1])
    with open('submit_temp.sub', 'w+') as ofile:
        ofile.write(sub_info)
    print(sub_info)
    os.system('sbatch submit_temp.sub')
    os.remove('submit_temp.sub')
