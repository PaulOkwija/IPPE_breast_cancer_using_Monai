import os
import pandas as pd

path = 'G:\WORK_2022\IPPE\Dataset_BUSI_with_GT'
classes = os.listdir('G:\WORK_2022\IPPE\Dataset_BUSI_with_GT')

data_dir = 'G:\WORK_2022\IPPE\Dataset_BUSI_with_GT'
class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
class_names

num_class = len(class_names)
data_files = [
    [
        x
        for x in os.listdir(os.path.join(data_dir, class_names[i]))
    ]
    for i in range(num_class)
]

data_files[0][:15]

# If there are masks, we separate them from the images in the folder
masks = [
    [
        x for x in data_files[i] if x.endswith('_mask.png') or x.endswith('_mask_1.png')
    ]
    for i in range(num_class)
]
mask_num = [len(masks[i]) for i in range(len(masks))]
print("Number of masks:",mask_num)

images = [
    [
        image for image in data_files[i] if image not in masks[i]
    ]
    for i in range(num_class)
]
image_num = [len(images[i]) for i in range(len(images))]
print("Number of masks:",image_num)

# Create the dataframe containing the respective class values for each image
data = {}
files = []
class_type = []
for num in range(len(class_names)):
    i = class_names[num]
    files = files + images[num]
    print(len(files))
    class_type = class_type + [i for file in images[num]]
    print(len(class_type))
data['images'] = files
data['class'] = class_type
df = pd.DataFrame(data)
df

pd.DataFrame.to_csv(df,'Breast_cancer_dataset.csv',index=False)