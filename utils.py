import os
import shutil
from PIL import Image, ImageShow
import cv2

print(os.getcwd())
mydir = 'train_bw'

def remove_all_files_in_testing():
    folderlist = [f for f in os.listdir(mydir) ]
    for f in folderlist:
        folder = mydir+'/'+f
        for files in os.listdir(folder):
            if(files.endswith('jpg')):
                os.remove(os.path.join(folder, files))

def move_files_in_training(percentage=20):
    mydir = 'training_images'
    mydirdst = 'testing_images'
    folderlist = [f for f in os.listdir(mydir)]
    for f in folderlist:
        folder = mydir + '/' + f
        folderdst = mydirdst + '/' + f
        path, dirs, files = next(os.walk(folder))
        file_count = len(files)
        file_pindah = (int)(file_count*percentage/100)
        print(folder+": {0} pindah {1}".format(file_count,file_pindah))
        for i in range (file_pindah):
            #print(folder+'/'+files[i])
            f_src = folder+'/'+files[i]
            f_dst = folderdst+'/'+files[i]
            shutil.move(f_src,f_dst)

def print_size(file, limitsize):
    im = Image.open(file)
    width, height = im.size
    if(width > limitsize or height > limitsize):
        print(file, " w : ", width, " h : ", height)

def check_size_image():
    folderlist = [f for f in os.listdir(mydir)]
    for f in folderlist:
        folder = mydir + '/' + f
        path, dirs, files = next(os.walk(folder))
        for file in files:
            f_src = folder + '/' + file
            print_size(f_src, 28)

def myresize(file_image, width):
    ori_img = cv2.imread(file_image)
    new_img = cv2.resize(ori_img,(width,width),interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(file_image,new_img)

def resize_image():
    folderlist = [f for f in os.listdir(mydir)]
    for f in folderlist:
        folder = mydir + '/' + f
        path, dirs, files = next(os.walk(folder))
        for file in files:
            f_src = folder + '/' + file
            myresize(f_src, 28)


def mysaveformatimg(file_image, format):
    new_img = Image.open(file_image)
    if file_image.endswith('jpg'):
        ls_new_image = file_image.split('.')
        file_image = ls_new_image[0]+'.'+format
    new_img.save(file_image,format)

def save_format_img():
    folderlist = [f for f in os.listdir(mydir)]
    for f in folderlist:
        folder = mydir + '/' + f
        path, dirs, files = next(os.walk(folder))
        for file in files:
            f_src = folder + '/' + file
            mysaveformatimg(f_src, 'png')

def convert_grayscale(image):
    #img = Image.open(image).convert('L')
    img = Image.open(image).convert('LA').convert('RGB')
    img.save(image)

def convert_into_grayscale():
    folderlist = [f for f in os.listdir(mydir)]
    for f in folderlist:
        folder = mydir + '/' + f
        path, dirs, files = next(os.walk(folder))
        for file in files:
            f_src = folder + '/' + file
            convert_grayscale(f_src)

# move_files_in_training()
# resize_image()
# check_size_image()
# mysaveformatimg('training_images/NU/NU_23.jpg','png')
# save_format_img()
# remove_all_files_in_testing()
convert_into_grayscale();
# convert_grayscale(mydir+'/0/0_2.jpg')