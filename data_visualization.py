import re
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

# log file path
log_file = 'output 2025-05-10.txt'

# 初始化存储数据的列表
# initialize a list to store data
data = []

# 正则表达式匹配所需信息
# regular expressions to match the required information
epoch_pattern = re.compile(r'epoch is (\d+), the whole loss is ([\d.]+)')
wer_pattern = re.compile(r'wer is ([\d.]+)')
sacc_pattern = re.compile(r'sacc is ([\d.]+)')

# 读取日志文件并提取信息
# read the log file and extract information
with open(log_file, 'r') as file:
    for line in file:
        epoch_match = epoch_pattern.search(line)
        wer_match = wer_pattern.search(line)
        sacc_match = sacc_pattern.search(line)

        if epoch_match:
            epoch = int(epoch_match.group(1))
            whole_loss = float(epoch_match.group(2))
        if wer_match:
            wer = float(wer_match.group(1))
        if sacc_match:
            sacc = float(sacc_match.group(1))
            # 将提取的数据存储到列表中
            data.append({'epoch': epoch, 'whole_loss': whole_loss, 'wer': wer, 'sacc': sacc})

# 将数据转换为 Pandas DataFrame
# convert the data to a Pandas DataFrame
df = pd.DataFrame(data)

# 保存到 CSV 文件
# save to CSV file
df.to_csv('filtered_data.csv', index=False)

# 打印 DataFrame
# print the DataFrame
print(df)


# 读取 CSV 文件
#  read the CSV file
df = pd.read_csv('filtered_data.csv')

# 绘制 Whole Loss 的变化趋势
# plot the trend of Whole Loss
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['whole_loss'], label='Whole Loss', color='blue', marker='o')
plt.title('Whole Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Whole Loss')
plt.legend()
plt.grid()
plt.show()

# 绘制 WER 的变化趋势
# plot the trend of WER
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['wer'], label='WER', color='green', marker='x')
plt.title('WER Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('WER')
plt.legend()
plt.grid()
plt.show()
