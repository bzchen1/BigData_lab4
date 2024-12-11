## 1.安装

下载`Spark(3.5.3版本)`，配置环境变量
```
#下载
wget https://dlcdn.apache.org/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz

# 配置环境变量
echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin' >> ~/.bashrc
source ~/.bashrc
```

安装`Pyspark`，配置环境变量
```
pip install pyspark

#配置环境变量
echo 'export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH' >> ~/.bashrc
echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc
source ~/.bashrc
```

## 2. 实验
### Task 1 : Spark RDD编程
#### 1. 查询特定⽇期的资⾦流⼊和流出情况
将读取内容转换为RDD，并提取`report_date`、`total_purchase_amt`、`row.total_redeem_amt`字段。然后对于相同的`report_date`，合并资金的流入、流出量。
#### 2.活跃⽤户分析
筛选出2014年8月的记录，提取user_id和report_date,生成键值对。接着进行去重(保证每天1个用户只被记作活跃1次)。然后进行合并，得到`< user_id , [ date1 , date2 , ... ] >`，最后计算`data`的长度，得到用户的活跃度，然后筛选长度大于5的用户。
运行结果：活跃用户数为12767
### Task 2: Spark SQL编程
#### 1. 按城市统计2014年3⽉1⽇的平均余额
将`user_balance`和`user_profile`两张表关联，筛选出`report_date=20140301`的记录。按照`city`分组，对每一个`city`的`tBalance`数组求平均，最后按照`avg_balance`降序排列。
#### 2. 统计每个城市总流量前3⾼的⽤户
将`user_balance`和`user_profile`两张表关联，筛选出`20140801<=report_date<=20140831`的记录。按照`city`和`user_id`进行分组。针对每个`city`，按照`user`的总流量降序排列，赋值`rank`。
最后输出每个城市`rank<=3`的用户。

### Task 3 Spark ML编程
思路：
- 读取表格，计算每日总购买赎回金额和平均余额，构建时间特征(星期几)和滞后特征(前7天的购买赎回数据)。合并基金收益率和拆借利率数据
- 使用Spark MLlib的LinearRegression算法，分别训练购买量和赎回量预测模型，`feature`包括:时间特征、历史滞后值、收益率指标和拆借利率
- 预测：使用8月最后一周的数据作为初始滞后特征，预测2014年9月的购买量、赎回量。
- 预测结果保存为tc_comp_predict_table.csv
评分：
![lab4报告-4.png]