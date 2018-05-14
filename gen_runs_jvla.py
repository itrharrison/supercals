import numpy as np
import glob

directory_list = glob.glob('/share/nas_mberc1/raid/harrison/superclass/jvla/*')
pointing_list = [directory.split('/')[-1] for directory in directory_list]

pointing_array = np.asarray(pointing_list)
pointing_chunks_array = np.array_split(pointing_array, 1)

for i,pointing_chunk in enumerate(pointing_chunks_array):
  
  ini_filename = 'inis/jvla_dr1-part{0}.ini'.format(i)
  ini_template = open('inis/jvla_dr1-part.template').read()
  pointing_string = ''
  for pointing in pointing_chunk: pointing_string+=','+str(pointing)
  pointing_string = pointing_string.lstrip(',')
  ini_file = ini_template.format(pointings=pointing_string)
  
  open(ini_filename, 'w').write(ini_file)
  
  launch_filename = 'launch/jvla_dr1-part{0}.sub'.format(i)
  launch_template = open('launch/jvla_dr1-part.template').read()
  launch_file = launch_template.format(part=i)
  
  open(launch_filename, 'w').write(launch_file)
  
  print('qsub {0}'.format(launch_filename))
