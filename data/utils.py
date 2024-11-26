import os

def kvasir_data_to_dict(folder):
    images = os.listdir(os.path.join(folder, 'images'))
    data_list_dict = []
    for i in images:
        try:     
            if os.path.exists(os.path.join(folder, 'masks', i)):
                data_dict = {'image': os.path.join(folder, 'images', i),
                             'label': os.path.join(folder, 'masks', i)}
                data_list_dict.append(data_dict)
        except Exception as e:
            print("Failed to find label file {} with exception {}".format(i, e))
    
    return data_list_dict