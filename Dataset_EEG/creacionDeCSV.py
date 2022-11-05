
# import OS module
import os
import shutil
import csv  

# Get the list of all files and directories
path = "C://Users//javie//Desktop//Dataset_EEG//Descanso"
images_descanso = os.listdir(path)

path = "C://Users//javie//Desktop//Dataset_EEG//M_RCH"
images_movimiento = os.listdir(path)

header = ['image_id', 'label']

train_images = []
test_images = []

train_descanso_len = int(len(images_descanso) * 0.8)

descanso_rows = []
movimiento_rows = []

for i in images_descanso:
  descanso_rows.append([i,'descanso'])

for i in images_movimiento:
  movimiento_rows.append([i,'movimiento'])


train_images.extend(descanso_rows[:train_descanso_len])
test_images.extend(descanso_rows[train_descanso_len:])

train_movimiento_len = int(len(images_movimiento) * 0.8)

train_images.extend(movimiento_rows[:train_movimiento_len])
test_images.extend(movimiento_rows[train_movimiento_len:])

print(train_images)
print(test_images)

for i in train_images:
  if i[1] == 'descanso':
    shutil.move("C://Users//javie//Desktop//Dataset_EEG//Descanso//" + i[0], "C://Users//javie//Desktop//Dataset_EEG//train_imgs//" + i[0])
  else:
    shutil.move("C://Users//javie//Desktop//Dataset_EEG//M_RCH//" + i[0], "C://Users//javie//Desktop//Dataset_EEG//train_imgs//" + i[0])

for i in test_images:
  if i[1] == 'descanso':
    shutil.move("C://Users//javie//Desktop//Dataset_EEG//Descanso//" + i[0], "C://Users//javie//Desktop//Dataset_EEG//test_imgs//" + i[0])
  else:
    shutil.move("C://Users//javie//Desktop//Dataset_EEG//M_RCH//" + i[0], "C://Users//javie//Desktop//Dataset_EEG//test_imgs//" + i[0])



with open('train_labels.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    for image in train_images:
      writer.writerow(image)

with open('test_labels.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    for image in test_images:
      writer.writerow(image)