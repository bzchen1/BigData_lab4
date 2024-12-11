from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# 创建SparkSession
spark = SparkSession.builder \
    .appName("UserBalanceAnalysis") \
    .getOrCreate()

# 读取CSV文件
balance_df = spark.read.csv("user_balance_table.csv", header=True, inferSchema=True)
profile_df = spark.read.csv("user_profile_table.csv", header=True, inferSchema=True)

# 将两个DataFrame注册为临时视图
balance_df.createOrReplaceTempView("user_balance")
profile_df.createOrReplaceTempView("user_profile")

# Task 1-1: 使用RDD方式计算每日资金流入流出
def daily_flow_rdd():
    # 转换为RDD并提取需要的字段
    rdd = balance_df.rdd.map(lambda row: (row.report_date, 
                                 (float(row.total_purchase_amt), 
                                  float(row.total_redeem_amt))))
    
    # 使用reduceByKey聚合
    result_rdd = rdd.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
                .sortByKey()
    
    # 保存到文件
    result_rows=result_rdd.map(lambda x: (x[0], x[1][0], x[1][1]))
    result_df=spark.createDataFrame(result_rows, ['report_date', 'total_purchase_amt', 'total_redeem_amt'])
    result_df.toPandas().to_csv('task1-1.csv', index=False)
        


# Task 1-2: 使用RDD方式统计活跃用户
def active_users_rdd():
    # 过滤2014年8月的数据并转换为RDD
    rdd = balance_df.rdd.filter(lambda row: str(row.report_date).startswith("201408")) \
                .map(lambda row: (row.user_id, row.report_date)) \
                .distinct() \
                .groupByKey() \
                .map(lambda x: (x[0], len(set(x[1])))) \
                .filter(lambda x: x[1] >= 5)
    
    # 计算满足条件的用户数
    active_count = rdd.count()
    print(f"活跃用户数: {active_count}")

# Task 2-1: 使用Spark SQL计算城市平均余额
def city_balance_sql():
    result_df = spark.sql("""
        SELECT up.city as city_id, 
               ROUND(AVG(ub.tBalance), 2) as avg_balance
        FROM user_balance ub
        JOIN user_profile up ON ub.user_id = up.user_id
        WHERE ub.report_date = 20140301
        GROUP BY up.city
        ORDER BY avg_balance DESC
    """)
    # 保存到文件
    result_df.toPandas().to_csv('task2-1.csv', index=False)

# Task 2-2: 使用Spark SQL统计城市TOP3用户
def city_top_users_sql():
    result_df = spark.sql("""
        WITH user_flow AS (
            SELECT 
                up.city as city_id,
                ub.user_id,
                SUM(ub.total_purchase_amt + ub.total_redeem_amt) as total_flow,
                RANK() OVER (PARTITION BY up.city 
                           ORDER BY SUM(ub.total_purchase_amt + ub.total_redeem_amt) DESC) as rank
            FROM user_balance ub
            JOIN user_profile up ON ub.user_id = up.user_id
            WHERE ub.report_date >= 20140801 AND ub.report_date <= 20140831
            GROUP BY up.city, ub.user_id
        )
        SELECT city_id, user_id, ROUND(total_flow, 2) as total_flow
        FROM user_flow
        WHERE rank <= 3
        ORDER BY city_id, rank
    """)
    result_df.toPandas().to_csv('task2-2.csv', index=False)



if __name__ == "__main__":
    print("===== Task 1-1: RDD方式计算每日资金流入流出(输出前20) =====")
    # daily_flow_rdd()
    
    print("\n===== Task 1-2: RDD方式统计活跃用户 =====")
    # active_users_rdd()
    
    print("\n===== Task 2-1: SQL方式计算城市平均余额 =====")
    city_balance_sql()
    
    print("\n===== Task 2-2: SQL方式统计城市TOP3用户 =====")
    city_top_users_sql()
    
    spark.stop()