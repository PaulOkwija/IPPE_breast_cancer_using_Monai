from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)

class BreastCancerDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


def path_list(path,mask_present,category):
    '''
    Takes in a path to a folder that has sub folders for each class of image data
    Returns:
    1. A list of lists for each class: index 0 corresponds to a list containing 
    paths for images belonging to the first alphabetical class.
    2. Number of classes
    3. List of classes
    '''

    data_dir = path
    class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
    class_names
    num_class = len(class_names)
    if mask_present:
        if category=='image':
            data_files = [
            [
                
                    x for x in data_files[i] if not (x.endswith('_mask.png') or x.endswith('_mask_1.png'))
                
            ]
            for i in range(num_class)
                        ]
        else:
            data_files = [
            [
                
                    x for x in data_files[i] if x.endswith('_mask.png') or x.endswith('_mask_1.png')
                
            ]
            for i in range(num_class)
                        ]
    
    else:
        data_files = [
            [
                x
                for x in os.listdir(os.path.join(data_dir, class_names[i]))
            ]
            for i in range(num_class)
        ]
    print(data_files[0][:15])
    return data_files, num_class, class_names


train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ]
)

val_transforms = Compose(
    [LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=num_class)])