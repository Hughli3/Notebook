# 分类算法学习
## 1.学习内容

> 使用算法
* 逻辑回归
* 决策树
* 随机森林
* 支持向量机

> 使用包
* 逻辑回归
+ R语言中基本函数 : glm()
*决策树及其可视化
+ rpart包
+ rpart.plot包
+ party 包
* 随机森林
+ randomForest包
* 支持向量机
+ e1071

## 2.数据准备
### 2.1数据来源

威斯康星州乳腺癌数据集是一个由逗号分隔的txt文件，可在UCI机器学习数据库 (http://archive.ics.uci.edu/ml)中找到。本数据集包含699个细针抽吸活检的样本单元，其中458个 15 (65.5%)为良性样本单元，241个(34.5%)为恶性样本单元。数据集中共有11个变量，表中未标明变量名。共有16个样本单元中有缺失数据并用问号(?)表示。 
 
 
### 2.2数据集变量情况
 
 > 数据集中包含的变量包括:
* ID
* 肿块厚度
* 细胞大小的均匀性  细胞形状的均匀性  边际附着力
* 单个上皮细胞大小 裸核
* 乏味染色体
* 正常核
* 有丝分裂
* 类别

第一个变量ID不纳入数据分析， _最后一个变量(类别)即输出变量(编码为良性=2，恶性=4)。_ 对于每一个样本来说，另外九个变量是与判别恶性肿瘤相关的细胞特征，并且得到了记录。
这些细胞特征得分为1(最接近良性)至10(最接近病变)之间的整数。 _任一变量都不能单独作 为判别良性或恶性的标准，建模的目的是找到九个细胞特征的某种组合，从而实现对恶性肿瘤的 准确预测。_


```r
loc <- "http://archive.ics.uci.edu/ml/machine-learning-databases/"  
#UCI机器学习数据库中的威斯康星州乳腺癌数据地址

ds <- "breast-cancer-wisconsin/breast-cancer-wisconsin.data"  
#地址下数据链接的名字，数据本身是txt格式

url <- paste(loc, ds, sep="") #组成url链接
breast <- read.table(url, sep=",", header=FALSE, na.strings="?") 
names(breast) <- c("ID", "clumpThickness", "sizeUniformity","shapeUniformity", "maginalAdhesion","singleEpithelialCellSize", "bareNuclei", "blandChromatin", "normalNucleoli", "mitosis", "class")

# ID
# 肿块厚度 "clumpThickness"
# 细胞大小的均匀性 "sizeUniformity"
# 细胞形状的均匀性 "shapeUniformity"
# 边际附着力 "maginalAdhesion"
# 单个上皮细胞大小 "singleEpithelialCellSize"
# 裸核 "bareNuclei"
# 乏味染色体 "blandChromatin"
# 正常核 "normalNucleoli"
# 有丝分裂 "mitosis"
# 类别:良性2，恶性4  "class"

df <- breast[-1] #去掉了数据中的ID列，breast[1]为选取第一列

df$class <- factor(df$class, levels=c(2,4),
                   labels=c("benign", "malignant"))
#注意本行不要重复运行，会覆盖数据df
#将数据集中的类别转为因子型，并给两种类型打上标签
#factor()函数将原来的数值型的向量转化为了factor类型。factor类型的向量中有Levels的概念。Levels就是factor中的所有元素的集合（没有重复）。可以发现Levels就是factor中元素排重后且字符化的结果！因为Levels的元素都是character。

set.seed(1234)
#随机种子 生产随机数过程中的随机因素

train <- sample(nrow(df), 0.7*nrow(df))    
 #nrow行数，在所有的数据中随机抽数，抽出其中的70%

df.train <- df[train,]      #训练集数据
df.validate <- df[-train,]      #测试集数据
table(df.train$class) 
table(df.validate$class)
#table 函数对应的就是统计学中的列联表，是一种记录频数的方法
```

## 3 逻辑回归
逻辑回归(logistic regression)是广义线性模型的一种，可根据一组数值变量预测二元输出。

```r
 fit.logit <- glm(class~., data=df.train, family=binomial())
#拟合逻辑回归

 summary(fit.logit)         #检查模型

Call:
glm(formula = class ~ ., family = binomial(), data = df.train)

Deviance Residuals: 
     Min        1Q    Median        3Q       Max  
-2.69040  -0.10584  -0.04834   0.01394   2.62641  

Coefficients:
                          Estimate Std. Error z value Pr(>|z|)    
(Intercept)              -11.42104    1.70065  -6.716 1.87e-11 ***
clumpThickness             0.62217    0.18141   3.430 0.000604 ***
sizeUniformity            -0.09447    0.32995  -0.286 0.774641    
shapeUniformity            0.34590    0.33822   1.023 0.306452    
maginalAdhesion            0.34038    0.15475   2.200 0.027835 *  
singleEpithelialCellSize   0.13920    0.18629   0.747 0.454944    
bareNuclei                 0.34587    0.11523   3.001 0.002687 ** 
blandChromatin             0.78192    0.24257   3.224 0.001266 ** 
normalNucleoli             0.08483    0.16218   0.523 0.600910    
mitosis                    0.72585    0.44606   1.627 0.103684    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 614.773  on 477  degrees of freedom
Residual deviance:  65.423  on 468  degrees of freedom
  (11 observations deleted due to missingness)
AIC: 85.423

Number of Fisher Scoring iterations: 8

prob <- predict(fit.logit, df.validate, type="response")
#predict()函数默认输出肿瘤为恶性的对数概率，制定参数type="response"即可得到预测肿瘤为恶性的概率。

logit.pred <- factor(prob > .5, levels=c(FALSE, TRUE),
labels=c("benign", "malignant"))
#对训练集集外样本单元分类

 logit.perf <- table(df.validate$class, logit.pred,
dnn=c("Actual", "Predicted"))
#the names to be given to the dimensions in the result (the dimnames names).
#评估预测准确性

 logit.perf
 #良性:benign 恶性:malignant

           Predicted
Actual      benign malignant
  benign       118         2
  malignant      4        76

 logit.fit.reduced <- step(fit.logit)
 #通过算AIC的方法，筛选变量获得精简模型

summary( logit.fit.reduced)

Call:
glm(formula = class ~ clumpThickness + shapeUniformity + maginalAdhesion + 
    bareNuclei + blandChromatin + normalNucleoli + mitosis, family = binomial(), 
    data = df.train)

Deviance Residuals: 
     Min        1Q    Median        3Q       Max  
-2.79835  -0.10622  -0.05886   0.01250   2.58072  

Coefficients:
                Estimate Std. Error z value Pr(>|z|)    
(Intercept)     -10.2455     1.4070  -7.282 3.29e-13 ***
clumpThickness    0.5133     0.1555   3.301 0.000962 ***
shapeUniformity   0.4303     0.2158   1.994 0.046184 *  
maginalAdhesion   0.3109     0.1406   2.211 0.027040 *  
bareNuclei        0.3353     0.1059   3.166 0.001547 ** 
blandChromatin    0.4392     0.2024   2.170 0.030030 *  
normalNucleoli    0.3013     0.1367   2.203 0.027570 *  
mitosis           0.6796     0.3982   1.707 0.087912 .  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 612.063  on 482  degrees of freedom
Residual deviance:  71.727  on 475  degrees of freedom
  (6 observations deleted due to missingness)
AIC: 87.727

Number of Fisher Scoring iterations: 8

```


# 4经典决策树
>概述

首先目前所提到的决策树主要分3种：ID3、C4.5、CART，其中ID3算法只能用于特征值为离散型的数据，C4.5和CART用的可以用于特征值为连续性的数据，但实现方式不同，C4.5所采用的方法与ID3相似，都采用以信息增益为标准判断节点，只不过C4.5多了一步将连续性变量认为划分为离散型。而CART则是使用gini系数来选择节点。

> CART算法步骤
* 通常分为两步建立回归树，最初生成一颗较大的树，然后通过统计估量删除底部的一些节点来对树进行修剪。这个过程的目的是防止过度拟合

1. 选定一个最佳预测变量将全部样本单元分为两类，实现两类中的纯度最大化（即一类中良性样本尽可能多，另一类中恶心样本尽可能多）。如果预测变量连续，则选择一个分割点进行分类，使得两类纯度最大化；如果预测变量为分类变量，则对各类别进行合并再分类。
2. 对每一个子类别继续执行步骤1
3. 重复1-2，直到子类别中所含的样本过少，或者没有分类法能将不纯度下降到一个给定的阈值以下最终集中的子类别终端节点根据每一个终端节点中样本的类别数众数来判别这一终端节点的所属类别
4. 对于一样本执行决策树，得到其终端节点，即可根据步骤3得到模型预测的所属类别。
- 不过上述算法通常会得到一颗过大的树，从而出现过拟合现象。结果就是，对于训练集外单元的分了性能较差。为解决这一问题，可采用10折交叉验证法选择预测误差最小的数。剪枝后的数即可用于预测

>CART算法原理
1. 数据准备工作

1. 


```r
library(rpart)
#加载包
set.seed(1234)
# 在构建树前可以先用rpart.control来控制树的大小
# 如ct <-rpart.control(xval = 10,minsplit = 20,cp = 0.1) #xval为设定10折交叉验证，minsplit设置最小分支节点数，cp是复杂度参数
dtree <- rpart(class ~ ., data=df.train, method="class",
parms=list(split="information"))
# 生成树
#缺失数据的默认处理办法是删除因变量缺失的观测而保留自变量缺失的观测。（na.action是处理缺失数据的方法参数）
------------------------------------------
 dtree$cptable

        CP nsplit rel error  xerror       xstd
1 0.800000      0   1.00000 1.00000 0.06484605
2 0.046875      1   0.20000 0.30625 0.04150018
3 0.012500      3   0.10625 0.20625 0.03467089
4 0.010000      4   0.09375 0.18125 0.03264401

#cpcp全称为complexity parameter，是复杂度参数，用于惩罚过大的树
#nsplit是树的大小，即分支数。n个分支的树将会有n+1个终端节点。
#rel error 是训练集中对应的各种误差
#xerror 是基于训练样本所得10折交叉验证误差
#xstd为交叉验证误差的标准差
-------------------------------------------
 summary(dtree)
 print(dtree)
 plotcp(dtree)
 #画出交叉验证误差与复杂度关系图，其中复杂度参数是x轴，交叉验证误差是y轴，虚线是基于一个标准差准则得到的上限。应该选虚线下最左侧cp值对应的树。
 
 dtree.pruned <- prune(dtree, cp=.0125)
 #剪枝
 #prune函数根据复杂度参数将最不重要的枝剪掉
 
 library(rpart.plot)
 prp(dtree.pruned, type = 2, extra = 104,
fallen.leaves = TRUE, main="Decision Tree")
 dtree.pred <- predict(dtree.pruned, df.validate, type="class")
 #对训练集外样本分类
 dtree.perf <- table(df.validate$class, dtree.pred,
dnn=c("Actual", "Predicted"))
 dtree.perf
```

> rpart包解释

1. rpart ( formula, data, w     eight s, subset, na. action = na. rpart, method, model= FALSE, x= FALSE,     y= TRU E, parms, cont rol, cost, . . . )
主要参数说明:

- fomula 回归方程形式: 例如 y~ x 1+ x     2+ x3。

- data 数据: 包含前面方程中变量的数据框( data     frame) 。

- na. action 缺失数据的处理办法:     默认办法是删除因变量缺失的观测而保留自变量缺失的观测。

- method 根据树末端的数据类型选择相应变量分割方法,本参数有四种取值: 连续型>anova; 离散型>class; 计数型( 泊松过程)>poisson; 生存分析型>exp。程序会根据因变量的类型自动选择方法, 但一般情况下较好还是指明本参数, 以便让程序清楚做哪一种树模型。

- parms 用来设置三个参数: 先验概率、损失矩阵、分类纯度的度量方法。anova没有参数；
poisson分割有一个参数，先验分布变异系数的比率，默认为1；
生存分布的参数和poisson一致；
对离散型，可以设置先验分布的分布的概率(prior)，损失矩阵(loss)，分类纯度(split）；
- 如: parms= list(propr=c(0.65,0.35),split = "information)
- priors必须为正值且和为1，loss必须对角为0且非对角为正数，split可以是gini（基尼系数）或者information（信息增益）；）

- control     控制每个节点上的最小样本量、交叉验证的次数、复杂性参量: 即cp: complexity pamemeter, 这个参数意味着对每一步拆分,     模型的拟合优度必须提高的程度, 等等。

剪枝: prune( ) 函数

2. prune(tree, . . . ) prune(     tree, cp, . . . )

- tree 一个回归树对象, 常是rpart( )     的结果对象。

- cp 复杂性参量, 指定剪枝采用的阈值。

- 通常分为两步建立回归树，最初生成一颗较大的树，然后通过统计估量删除底部的一些节点来对树进行修剪。这个过程的目的是防止过度拟合

- 使用rpart函数构建树的过程中，当给定条件满足时构建过程就停止。偏差的减少小于某一个给定界限值、节点中的样本数量小于某个给定界限、树的深度大于一个给定的界限，上面三个界限分别由rpart()函数的三个参数(cp、minsplit、maxdepth)确定，默认值是0.01、20和30。如果要避免树的过度拟合问题，就要经常检查这些默认值的有效性，这可以通过对得到的树采取事后修剪的过程来实现。

- 选择树的方法一般有两种，一种是最小化交叉验证的相对方差（xerror）。另外一种是在剪枝理论中,     比较著名的规则就是1- SE( 1标准差) 规则, 其意思是: 首先要保证预测误差( 通过交叉验证获得, 在程序中表示为xerror) 尽量小,     但不一定要取最小值, 而是允许它在“最小的误差”一个相应标准差0、的范围内, 然后在此范围内选取尽量小的复杂性参量,     进而以它为依据进行剪枝。这个规则体现了兼顾树的规模( 复杂性) 和误差大小的思想, 因为一般说来, 随着拆分的增多, 复杂性参量会单调下降(     纯度越来越高) , 但是预测误差则会先降后升, 这样, 就无法使复杂性和误差同时降到较低,因此允许误差可以在一个标准差内波动

# 条件决策树
条件推断树与传统决策树类似，但变量和分割的选取是基于显著性检验的，而不是纯净度或同质性的度量。显著性检验是置换检验。

## 算法
1. 对输出变量与每个预测变量间的关系计算p值
1. 选取p值最小的变量
1. 在因变量与被选中的变量间尝试所有可能的二元分割（通过排列检验），并选取最显著的分割。
1. 将数据集分成两群，并对每个子群重复上述步骤
1. 重复直至所有分割都不显著或已到达最小节点为止

```r
library(party) 
fit.ctree <- ctree(class~., data=df.train) 
plot(fit.ctree, main="Conditional Inference Tree") 
 
 ctree.pred <- predict(fit.ctree, df.validate, type="response") 
 ctree.perf <- table(df.validate$class, ctree.pred,
                     dnn=c("Actual", "Predicted")) 
 ctree.perf 
 

```

# 随机森林
在随机森林中，我们同时生成了多个预测模型，并将模型的结果汇总以提升分类准确性。
## 算法
- 假设训练集中共有N个样本单元，M个变量，则随机森林算法如下：
1. 从训练机中随机又放回地抽取N个样本单元，生成大量决策树。
    - 相当于在所有的数据中抽取一些行（有放回）出来，这种做法可以防止过拟合
    - 在R中的随机森林默认在每个节点抽取了sqrt(M)个变量
1. 在每一个节点随机抽取m<M个变量，将其作为分割该节点的候选变量。每个节点处的变量数应该一致
    - 在完成行抽样后进行列抽样，相当于抽出这些样本的某几个特征去构建树，这样构建的树都是专家树（因为只针对于某几个特征去判断）
1. 以之上两部抽样得到的数据完整生成所有决策树（每一次的抽样都会构建出一棵树），无需剪枝（最小节点为1） ，某些树可能仅仅有一个节点，这是由于抽样的数据量决定的。
1. 终端节点的所属类别由节点对应的众数类别决定
1. 对于新的观测点，用所有的树对其进行分类，其类别由多数决定原则生成。

> Tips
- 在R中randomForest包基于传统决策树（CART）去构建随机森林
- 在party包中的cforest()函数则用条件推断树生成随机森林
```r
 library(randomForest) 
 set.seed(1234) 
 fit.forest <- randomForest(class~., data=df.train,  
na.action=na.roughfix, importance=TRUE) 
# na.action=na.roughfix将数值变量中的缺失值替换成了队列中的中位数，将类别变量中的缺失值替换为对应列的众数类，如果有多个众数则随机选择一个。
# 通过设置information度量变量的重要性，可通过importance函数输出
 fit.forest 
 
 
importance(fit.forest, type=2) 
#tyoe=2 参数的道德变量相对重要性就是分割该变量时节点不纯度
 forest.pred <- predict(fit.forest, df.validate) 
 forest.perf <- table(df.validate$class, forest.pred,
 dnn=c("Actual", "Predicted")) 
 forest.perf 
```

```r
 library(e1071) 
 set.seed(1234) 
 fit.svm <- svm(class~., data=df.train) 
 fit.svm  

Call: svm(formula = class ~ ., data = df.train)   
Parameters:     SVM-Type: C-classification  SVM-Kernel: radial        cost: 1       gamma: 0.1111  

Number of Support Vectors: 76  
 svm.pred <- predict(fit.svm, na.omit(df.validate)) svm.perf <- table(na.omit(df.validate)$class,                     svm.pred, dnn=c("Actual", "Predicted")) 
  svm.perf  

           Predicted Actual      benign malignant   benign       116         4   malignant      3        77 
```

```r
sqldf("select * from 
(select prod_inst_id as id,
acc_num as phone_number,
month_id ,
call_duration-b_call_duration as call_duration_time,
call_type_b-b_c_call_type_b as call_time,
call_type_Z-b_c_call_type_Z as called_time,
lfee-b_a_lfee ,
call_duration_s-b_n_call_duration_s net_time,
send_bytes-b_n_send_bytes send_types,
prod_inst_type
from
(
select a.*,
b.prod_inst_id as b_prod_inst_id ,
b.acc_num as  b_pacc_num,
b.month_id as b_b_month_id,
b.call_duration as b_call_duration,
b.call_type_b as  b_c_call_type_b,
b.call_type_Z as b_c_call_type_Z,
b.lfee as b_a_lfee,
b.call_duration_s as b_n_call_duration_s,
b.send_bytes as b_n_send_bytes,
b.prod_inst_type as b_p_prod_inst_type
from 
(select * from data8
where prov_id is not null 
and call_duration <> NA)a
inner join
(select * from data7
where prov_id is not null)b
  on a.prod_inst_id = b.prod_inst_id
  and a.prod_inst_type=b.prod_inst_type

  )c  ) d
  where d.call_duration_time is not null")
```