import cv2


def augment(x_train, x_target, sensor):
    print('...flip & rotate...')
    new_x_train = []
    new_x_label = []
    for idx,i in enumerate(x_train):
        # width, height = i.shape
        if sensor == 'Orcathus':
            i = cv2.resize(i,(300, 300))
        h = cv2.flip(i,1)
        h90 = cv2.rotate(h, cv2.ROTATE_90_CLOCKWISE)
        h180 = cv2.rotate(h, cv2.ROTATE_180)
        h270 = cv2.rotate(h, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        x90 = cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE)
        x180 = cv2.rotate(i, cv2.ROTATE_180)
        x270 = cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        new_x_train.append(i)
        new_x_train.append(x90)
        new_x_train.append(x180)
        new_x_train.append(x270)
        new_x_train.append(h)
        new_x_train.append(h90)
        new_x_train.append(h180)
        new_x_train.append(h270)
        
        for i in range(8):
            new_x_label.append(x_target[idx])
    return new_x_train, new_x_label

def augment_test(x_test):
    print('...flip & rotate...')
    new_x_test = []
    for idx,i in enumerate(x_test):
        i = cv2.resize(i,(300, 300))
        new_x_test.append(i)
        
    return new_x_test
        