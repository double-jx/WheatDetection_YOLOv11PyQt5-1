#coding:utf-8

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 使用的模型路径
model_path = 'models/best.pt'

# 数据集类别与名称
names = {
    0: 'wheat-stemrust',
    1: 'wheat-healthy',
    2: 'wheat-smut',
    3: 'wheat-yellowrust',
    4: 'wheat-brownrust'
}

# 数据集类别中文
CH_names = ['秆锈病', '健康', '散黑穗病', '黄锈病', '叶锈病']

# csv文件保存路径
csv_save_path = 'save_data/save_detect_data.csv'