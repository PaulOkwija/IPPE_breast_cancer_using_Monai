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
