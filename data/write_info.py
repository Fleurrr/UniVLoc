import os

dir1 = './training_0503/'

def check_dir(path):
  count = 0
  for item in os.listdir(path):
    if os.path.isdir(os.path.join(path, item)):
      count += 1
  return count

out_path = './data_info.txt'
with open(out_path, 'a') as file:
  dir1_places = os.listdir(dir1)
  dir1_places.sort()
  for dir1_place in dir1_places:
    file_path = os.path.join(dir1, dir1_place, 'others/sessions')
    count = check_dir(file_path)
    content = dir1_place + ' ' + str(count) +'\n'
    file.write(content)