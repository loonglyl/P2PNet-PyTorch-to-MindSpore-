import scipy
import shutil
'''
1.读取mat文件改写为每行为一个x, y坐标值的txt文件，并重命名序号防止partA和partB的图片序号重合导致数据覆盖丢失，
保存到DATA_ROOT中的对应数据集的annotations中。
2.读取jpg文件并改名保存到DATA_ROOT中对应数据集的images中。
'''

# 训练集图像绝对地址与目标地址
train_image_partA = r'D:\Python\programs\Dataset\ShanghaiTech\part_A\train_data\images'
train_image_partB = r'D:\Python\programs\Dataset\ShanghaiTech\part_B\train_data\images'
train_image_destination = r'D:\Python\programs\Projects\P2PNet\CrowdCounting-P2PNet-main\DATA_ROOT\train\Images'

# 测试集图像绝对地址与目标地址
test_image_partA = r'D:\Python\programs\Dataset\ShanghaiTech\part_A\test_data\images'
test_image_partB = r'D:\Python\programs\Dataset\ShanghaiTech\part_B\test_data\images'
test_image_destination = r'D:\Python\programs\Projects\P2PNet\CrowdCounting-P2PNet-main\DATA_ROOT\test\Images'

# 训练集标注数据绝对地址与目标地址
train_ground_partA = r'D:\Python\programs\Dataset\ShanghaiTech\part_A\train_data\ground-truth'
train_ground_partB = r'D:\Python\programs\Dataset\ShanghaiTech\part_B\train_data\ground-truth'
train_ground_destination = r'D:\Python\programs\Projects\P2PNet\CrowdCounting-P2PNet-main\DATA_ROOT\train\Annotations'

# 测试集标注数据绝对地址与目标地址
test_ground_partA = r'D:\Python\programs\Dataset\ShanghaiTech\part_A\test_data\ground-truth'
test_ground_partB = r'D:\Python\programs\Dataset\ShanghaiTech\part_B\test_data\ground-truth'
test_ground_destination = r'D:\Python\programs\Projects\P2PNet\CrowdCounting-P2PNet-main\DATA_ROOT\test\Annotations'

train_num_of_partA = 300
train_num_of_partB = 400
test_num_of_partA = 182
test_num_of_partB = 316


def mat_to_txt(src, dst, start_number, num):
    for i in range(start_number, start_number + num):
        data = scipy.io.loadmat(src+r'\GT_IMG_'+str(i - start_number + 1)+'.mat')  # 读取mat文件,字典格式
        # print(data['image_info'][0][0][0][0][0])
        data = data['image_info'][0][0][0][0][0]  # 获取标注数据数组
        with open(dst+r'\GT_IMG_'+str(i)+'.txt', 'a') as f:
            for x in range(len(data)):  # 一维的长度
                f.write(str(data[x][0])+' '+str(data[x][1])+'\n')


def move_images(src, dst, start_number, num):
    for i in range(start_number, start_number + num):
        shutil.copy(src+r'\IMG_'+str(i - start_number + 1)+'.jpg', dst+r'\IMG_'+str(i)+'.jpg')


# # 把训练集和测试集的A、B部分的标注数据都写入Annotations中的txt文件中
# mat_to_txt(train_ground_partA, train_ground_destination, start_number=1, num=train_num_of_partA)  # 从1开始
# mat_to_txt(train_ground_partB, train_ground_destination, start_number=1 + train_num_of_partA, num=train_num_of_partB)
#
# mat_to_txt(test_ground_partA, test_ground_destination, start_number=1, num=test_num_of_partA)  # 从1开始
# mat_to_txt(test_ground_partB, test_ground_destination, start_number=test_num_of_partA + 1, num=test_num_of_partB)
#
# # 拷贝图像并重新分配序号
# move_images(train_image_partA, train_image_destination, start_number=1, num=train_num_of_partA)
# move_images(train_image_partB, train_image_destination, start_number=1 + train_num_of_partA, num=train_num_of_partB)
#
# move_images(test_image_partA, test_image_destination, start_number=1, num=test_num_of_partA)
# move_images(test_image_partB, test_image_destination, start_number=1 + test_num_of_partA, num=test_num_of_partB)

# 写train.list和test.list文件
# with open(r'D:\Python\programs\Projects\P2PNet\CrowdCounting-P2PNet-main\DATA_ROOT\train.txt', 'w') as f:
#     for i in range(1, train_num_of_partA + train_num_of_partB + 1):
#         f.write(r'train\Images\IMG_'+str(i)+'.jpg'+' '+r'train\Annotations\GT_IMG_'+str(i)+'.txt'+'\n')
#
# with open(r'D:\Python\programs\Projects\P2PNet\CrowdCounting-P2PNet-main\DATA_ROOT\test.txt', 'w') as f:
#     for i in range(1, test_num_of_partA + test_num_of_partB + 1):
#         f.write(r'test\Images\IMG_'+str(i)+'.jpg'+' '+r'test\Annotations\GT_IMG_'+str(i)+'.txt'+'\n')
