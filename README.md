# seq2seq-attention
# Data 
data: 	
         data\processed  语料库及字典   
data\samples:   
                human.txt 为随机选择用来测试预测效果的诗句   
                default.txt   预测生成诗句   
model:     保存训练完成后模型   

# Code
model_original.py:seq2seq+attention 模型   
Train.py   训练函数   
predict.py: 预测函数   
main.py: 用户交互界面   
Plan.py  用来生成训练语句的关键词   
Word2vec.py 通过语料库生成word embeding   

# 运行流程： 
1.data_utils.py  对语料库进行清洗，并生成固定的格式的训练数据   
2.Train.py      通过对数据进行训练，生成模型   
3.Predict.py     随机选取训练诗句作为预测样本，并生成预测诗句   
