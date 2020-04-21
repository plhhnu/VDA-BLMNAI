# VDA-BLMNAL

VDI-BLMNII将病毒相似性、药物相似性和病毒-药物关联矩阵（txt）作为输入文件，.txt文件格式用于存储所有新的病毒与药物关联得分。
命令：
Python3 VDI.py [-PARAMS PARAM_VALUE]
参数：
-method 方法名字
-dataset数据集的名字
-predict-num 默认为0，一个正数表示top-N新的VDIs
示例：
Python3 VDI.py --method="blmnii" --dataset="dv" --predict-num=100
python3 VDI.py --method="blmnii" --dataset="dv"
