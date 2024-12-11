from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col, dayofweek, avg, sum, lag, lit,to_date
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import pandas as pd

# 创建SparkSession
spark = SparkSession.builder \
    .appName("Purchase_Redeem_Prediction") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# 读取所有数据表
balance_df = spark.read.csv("/home/lab4/data/user_balance_table.csv", header=True, inferSchema=True)
interest_df = spark.read.csv("/home/lab4/data/mfd_day_share_interest.csv", header=True, inferSchema=True)
shibor_df = spark.read.csv("/home/lab4/data/mfd_bank_shibor.csv", header=True, inferSchema=True)

def prepare_features():
    # 先按日期汇总购买和赎回金额
    daily_sum = balance_df.groupBy("report_date") \
        .agg(
            sum("total_purchase_amt").alias("total_purchase"),
            sum("total_redeem_amt").alias("total_redeem"),
            avg("tBalance").alias("avg_balance")
        )
    
    # 添加日期列
    daily_sum = daily_sum.withColumn(
        "date",
        to_date(col("report_date").cast("string"), "yyyyMMdd")
    )
    
    # 添加星期几特征
    daily_sum = daily_sum.withColumn(
        "day_of_week", 
        dayofweek("date")
    )
    
    # 添加滞后特征
    window_spec = Window.orderBy("date")
    for i in range(1, 8):
        daily_sum = daily_sum \
            .withColumn(f"purchase_lag_{i}", lag("total_purchase", i).over(window_spec)) \
            .withColumn(f"redeem_lag_{i}", lag("total_redeem", i).over(window_spec))
    
    # 合并收益率数据
    if interest_df is not None:
        interest_df_with_date = interest_df.withColumn(
            "join_date", 
            to_date(col("mfd_date").cast("string"), "yyyyMMdd")
        )
        daily_sum = daily_sum.join(
            interest_df_with_date,
            daily_sum.date == interest_df_with_date.join_date,
            "left"
        ).drop("join_date")
        
    # 合并拆借利率数据
    if shibor_df is not None:
        shibor_df_with_date = shibor_df.withColumn(
            "join_date", 
            to_date(col("mfd_date").cast("string"), "yyyyMMdd")
        )
        daily_sum = daily_sum.join(
            shibor_df_with_date,
            daily_sum.date == shibor_df_with_date.join_date,
            "left"
        ).drop("join_date")
    
    # 填充空值并删除不需要的列
    return daily_sum.drop("mfd_date").na.fill(0)

def prepare_train_test_data(df):
    # 选择特征列
    feature_cols = [
        "day_of_week",
        "avg_balance",
        "mfd_daily_yield",
        "mfd_7daily_yield",
        "Interest_O_N",
        "Interest_1_W",
        "Interest_1_M"
    ] + [f"purchase_lag_{i}" for i in range(1, 8)] + [f"redeem_lag_{i}" for i in range(1, 8)]
    
    # 创建特征向量
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    

    # 准备数据
    prepared_df = df.select(["report_date"] + feature_cols + ["total_purchase", "total_redeem"]) \
                   .na.fill(0)
    
    # 转换特征
    vectorized_df = assembler.transform(prepared_df)
    return (
        vectorized_df.select("report_date", "features", col("total_purchase").alias("label")),
        vectorized_df.select("report_date", "features", col("total_redeem").alias("label"))
    )

def train_models(purchase_data, redeem_data):
    # 训练购买量预测模型
    purchase_model = LinearRegression(maxIter=100)
    purchase_model = purchase_model.fit(purchase_data)
    
    # 训练赎回量预测模型
    redeem_model = LinearRegression(maxIter=100)
    redeem_model = redeem_model.fit(redeem_data)
    
    return purchase_model, redeem_model

def prepare_prediction_features(history_df):
    """为预测数据准备特征"""
	# 创建基础DataFrame
    dates = [(datetime(2014, 9, 1) + timedelta(days=x)).strftime("%Y%m%d") for x in range(30)]
    pred_df = spark.createDataFrame([(date,) for date in dates], ["report_date"])
    
    # 添加日期列和星期几特征
    pred_df = pred_df.withColumn(
        "date",
        to_date(col("report_date").cast("string"), "yyyyMMdd")
    ).withColumn(
        "day_of_week",
        dayofweek("date")
    )
    
    # 添加收益率数据
    if interest_df is not None:
        # 使用8月份最后一周的平均收益率
        last_week_interest = interest_df.filter(
            (col("mfd_date") >= 20140825) & 
            (col("mfd_date") <= 20140831)
        ).agg(
            avg("mfd_daily_yield").alias("mfd_daily_yield"),
            avg("mfd_7daily_yield").alias("mfd_7daily_yield")
        ).collect()[0]
        
        pred_df = pred_df.withColumn("mfd_daily_yield", lit(last_week_interest["mfd_daily_yield"])) \
                        .withColumn("mfd_7daily_yield", lit(last_week_interest["mfd_7daily_yield"]))
    
    # 添加拆借利率数据
    if shibor_df is not None:
        # 使用8月份最后一周的平均利率
        last_week_shibor = shibor_df.filter(
            (col("mfd_date") >= 20140825) & 
            (col("mfd_date") <= 20140831)
        ).agg(
            avg("Interest_O_N").alias("Interest_O_N"),
            avg("Interest_1_W").alias("Interest_1_W"),
            avg("Interest_1_M").alias("Interest_1_M")
        ).collect()[0]
        
        pred_df = pred_df.withColumn("Interest_O_N", lit(last_week_shibor["Interest_O_N"])) \
                        .withColumn("Interest_1_W", lit(last_week_shibor["Interest_1_W"])) \
                        .withColumn("Interest_1_M", lit(last_week_shibor["Interest_1_M"]))
    
    # 使用8月份最后7天的数据作为滞后特征的初始值
    last_week_data = history_df.orderBy(col("date").desc()).limit(7) \
                              .select("total_purchase", "total_redeem") \
                              .collect()
    
    for i in range(1, 8):
        if i <= len(last_week_data):
            pred_df = pred_df.withColumn(f"purchase_lag_{i}", 
                                       lit(float(last_week_data[i-1]["total_purchase"])))
            pred_df = pred_df.withColumn(f"redeem_lag_{i}", 
                                       lit(float(last_week_data[i-1]["total_redeem"])))
        else:
            pred_df = pred_df.withColumn(f"purchase_lag_{i}", lit(0.0))
            pred_df = pred_df.withColumn(f"redeem_lag_{i}", lit(0.0))
    
    # 添加平均余额（使用8月份的平均值）
    avg_balance = history_df.agg(avg("avg_balance")).collect()[0][0]
    pred_df = pred_df.withColumn("avg_balance", lit(avg_balance))
    
    return pred_df.na.fill(0)

# 添加生成预测的函数
def generate_predictions(purchase_model, redeem_model, feature_df):
    # 准备预测特征
    pred_features_df = prepare_prediction_features(feature_df)
    
    # 向量化特征
    feature_cols = [
        "day_of_week",
        "avg_balance",
        "mfd_daily_yield",
        "mfd_7daily_yield",
        "Interest_O_N",
        "Interest_1_W",
        "Interest_1_M"
    ] + [f"purchase_lag_{i}" for i in range(1, 8)] + [f"redeem_lag_{i}" for i in range(1, 8)]
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    pred_vector_df = assembler.transform(pred_features_df)
    
    # 生成预测
    purchase_predictions = purchase_model.transform(pred_vector_df)
    redeem_predictions = redeem_model.transform(pred_vector_df)
    
    # 合并预测结果
    predictions = purchase_predictions.select(
        col("report_date"),
        col("prediction").alias("total_purchase_amt")
    ).join(
        redeem_predictions.select(
            col("report_date"),
            col("prediction").alias("total_redeem_amt")
        ),
        "report_date"
    )
    
    # 转换为Pandas DataFrame并格式化
    pd_predictions = predictions.toPandas()
    pd_predictions['report_date'] = pd_predictions['report_date'].astype(str)
    pd_predictions[['total_purchase_amt', 'total_redeem_amt']] = pd_predictions[
        ['total_purchase_amt', 'total_redeem_amt']
    ].round(2)
    
    return pd_predictions

def main():
    # 准备特征
    print("准备特征...")
    feature_df = prepare_features()
    
    # 准备训练数据
    print("准备训练数据...")
    purchase_data, redeem_data = prepare_train_test_data(feature_df)
    
    # 训练模型
    print("训练模型...")
    purchase_model, redeem_model = train_models(purchase_data, redeem_data)
    
    # 生成预测
    print("生成预测...")
    predictions_df = generate_predictions(purchase_model, redeem_model, feature_df)
    
    # 保存预测结果
    predictions_df.to_csv('tc_comp_predict_table.csv', index=False)
    print("预测结果已保存到tc_comp_predict_table.csv")

if __name__ == "__main__":
    main()
    spark.stop()