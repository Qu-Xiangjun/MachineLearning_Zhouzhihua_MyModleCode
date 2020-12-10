"""
@time: 2020-10-15
@author: Qu Xiangjun 20186471
@describe: 测试自定义的决策树分类算法对西瓜数据集2.0(周志华《机器学习》)二分类
"""
import pandas as pd
import json
from Decision_Tree import Create_Decision_Tree, StoreTree,ReadTree,dtClassify
from draw_decision_tree import create_plot


def load_dataset():
    """
    从asv表格导入数据
    @return x 训练数据 shape=[6,17] ,
            y 训练数据的结果 shape =[1,17],
            attribute 属性list，包含字典{属性名称：取值} 
            eg:[{'色泽': ['浅白', '青绿', '乌黑']}, {'根蒂': ['蜷缩', '硬挺', '稍蜷']}]
    """
    filename = 'E:\\Study\\Term5\\Merchine Learning\\Lab\\Lab02\\xigua2.0.csv'
    data = pd.read_excel(filename)
    # 7行17列的矩阵，第一行为色泽，第二行为根蒂，
    # 第三行为敲声，第四行为纹理，第五行为脐部，第柳行为触感
    x = [list(data[u'色泽']), list(data[u'根蒂']),
        list(data[u'敲声']), list(data[u'纹理']),
        list(data[u'脐部']), list(data[u'触感'])]
    # 1行17列，好瓜结果，1为好瓜，0为坏瓜
    y = list(data[u'好瓜'])
    # 属性：取值
    attribute = []
    attribute_name = ['色泽','根蒂','敲声','纹理','脐部','触感']
    for i in range(len(x)):
        label = set(x[i])
        lable_list = list(label)
        dic = {}
        dic[attribute_name[i]] = lable_list
        attribute.append(dic)
    
    return x,y,attribute

if __name__ == "__main__":
    x,y,attribute = load_dataset()
    # ans = Create_Decision_Tree(x,y,attribute,method='id3')
    # ans = Create_Decision_Tree(x,y,attribute,method='c4.5')
    ans = Create_Decision_Tree(x,y,attribute,method='cart')
    create_plot(ans)
    print(json.dumps(ans, ensure_ascii=False)) 
    # 模型存储为json文件
    # StoreTree(ans,'E:\\Study\\Term5\\Merchine Learning\\Lab\\Lab02\\tree_id3.json')  
    # StoreTree(ans,'E:\\Study\\Term5\\Merchine Learning\\Lab\\Lab02\\tree_c4.5.json')  
    # StoreTree(ans,'C:\\Users\\49393\\Desktop\\ML_pre\\实验2.1-屈湘钧-20186471-2018计科卓越\\tree_cart.json')  
    
    testAns = dtClassify(ans,['色泽','根蒂','敲声','纹理','脐部','触感'], x)
    print("y:",y)
    count = 0 # 正确率
    print("test:",testAns)
    for i in range(len(y)):
        if(y[i] == testAns[i]):
            count += 1
    print("正确率：%.2f"%(count/len(y)*100.0))