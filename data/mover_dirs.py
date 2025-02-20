import os


data_dir = './training_0503'
data_lists = os.listdir(data_dir)
data_lists.sort()
for data in data_lists:
    session_dir = os.path.join(data_dir, data, 'others/sessions/sessions/')
    command = 'mv ' + session_dir + '*  ' + os.path.join(data_dir, data, 'others/sessions')
    print(command)
    os.system(command)