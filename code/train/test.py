import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# 将图片编码为rle格式
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 将rle格式进行解码为图片
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

train_mask = pd.read_csv('../../user_data/data/train_mask.csv', sep = '\t', names = ['name', 'mask'])
test_mask=pd.read_csv('../../user_data/data/test_a_samplesubmit.csv', sep = '\t', names = ['name', 'mask'])
#读取第一张图，并将对应的rle解码为mask矩阵
img = cv2.imread("../../user_data/data/train/" + train_mask['name'].iloc[0])
mask = rle_decode(train_mask['mask'].iloc[0])
print(rle_encode(mask) == train_mask['mask'].iloc[0])  #true
img=cv2.imread("../../user_data/data/train/0AK9VMA5R5.jpg")
mask=rle_decode(train_mask['mask'].iloc[0])
print(rle_encode(mask) == train_mask['mask'].iloc[0])
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(mask,cmap='gray')
plt.show()
print(img)
noBuild = 0
BuildArea = 0
for idx in np.arange(train_mask.shape[0]):
    if pd.isnull(train_mask['mask'].iloc[idx]):
        noBuild += 1
    else:
        mask = rle_decode(train_mask['mask'].iloc[idx])
        BuildArea += np.sum(mask)

#统计所有图片中建筑物区域平均区域大小
meanBuildArea = BuildArea / (train_mask.shape[0]-noBuild)

#统计所有图片中建筑物像素占所有像素的比例
buildPixPerc = BuildArea / (train_mask.shape[0]*mask.shape[0]*mask.shape[1])

#统计所有图片整图中没有任何建筑物像素占所有训练集图片的比例
noBuildPerc = noBuild/train_mask.shape[0]


A=['have buildings','nobuildings']

T=[train_mask.shape[0],noBuild]

plt.pie(T,labels=A,autopct='%1.1f%%')

# plt.title('')
plt.show()

name_list = ['train','test']
num_list = [train_mask.shape[0],test_mask.shape[0]]
plt.bar(range(len(num_list)), num_list,tick_label=name_list)
plt.show()

print("The percentage of image containing no buildings: %.4f" % noBuildPerc)
print("The percentage of pixels of building: %.4f" % buildPixPerc)
print("The mean area of buildings in an image: %d" % meanBuildArea)
print("trainimage: ",train_mask.shape[0] )
print("testimage: ",test_mask.shape[0])
