## Chinese Article Generate 2019-1

#### 1.preprocess

prepare() 将 txt 数据处理为 (poet, title, text) 的三元组，保存为 csv 格式

建立 poetry 字典，实现作者、标题、正文的两层映射，check() 检查索引连续性

#### 2.retrieve

输入作者与关键词，查找此作者所有包含关键词的标题及正文

#### 3.explore

统计作者、标题、正文词汇、正文长度的频率，条形图可视化

#### 4.represent

add_flag() 添加控制符，word2vec() 按字训练词向量、构造 embed_mat

shift() 分别删去 bos、eos 得到 sent、label，align() 分别截取或填充为定长序列

#### 5.build

通过 trm 构建语言生成模型，对解码器词特征 x 进行多头线性映射

得到 q、k、v，使用 mask 点积注意力得到语境向量 c、再线性映射进行降维

#### 6.generate

对概率前 5 的输出归一化后采样，逗号、句号为最大概率则直接返回

生成结束符或长度大于 max_len 时停止、小于 min_len 则采样直到非结束符