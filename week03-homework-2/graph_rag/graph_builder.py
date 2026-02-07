"""
构建 Neo4j 知识图谱

从 CSV 文件读取股权数据并构建知识图谱
"""

import pandas as pd
from neo4j import GraphDatabase
import os
from pathlib import Path


def build_graph(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password123",
    neo4j_database: str = "neo4j",
    csv_path: str = None
):
    """
    从 CSV 文件读取数据并构建 Neo4j 知识图谱

    Args:
        neo4j_uri: Neo4j 连接 URI
        neo4j_user: Neo4j 用户名
        neo4j_password: Neo4j 密码
        neo4j_database: Neo4j 数据库名称
        csv_path: CSV 文件路径，默认为 data/shareholders.csv
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / "data" / "shareholders.csv"

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[ERROR] CSV 文件不存在: {csv_path}")
        return

    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password)
    )

    try:
        df = pd.read_csv(csv_path)
        print(f"[INFO] 读取到 {len(df)} 条股权记录")

        with driver.session(database=neo4j_database) as session:
            # 清空数据库以避免重复创建
            print("[INFO] 正在清空现有图谱数据...")
            session.run("MATCH (n) DETACH DELETE n")

            print("[INFO] 正在创建节点和关系...")
            # 使用 UNWIND 批量创建，效率更高
            # 1. 创建所有公司和股东节点
            query_create_nodes = """
            UNWIND $rows AS row
            MERGE (c:Entity {name: row.company_name})
            ON CREATE SET c.type = '公司'
            MERGE (s:Entity {name: row.shareholder_name})
            ON CREATE SET s.type = row.shareholder_type
            """
            session.run(query_create_nodes, rows=df.to_dict('records'))

            # 2. 创建持股关系
            query_create_rels = """
            UNWIND $rows AS row
            MATCH (shareholder:Entity {name: row.shareholder_name})
            MATCH (company:Entity {name: row.company_name})
            MERGE (shareholder)-[r:HOLDS_SHARES_IN]->(company)
            SET r.share_percentage = toFloat(row.share_percentage)
            """
            session.run(query_create_rels, rows=df.to_dict('records'))

            print("[INFO] 图谱节点和关系创建完成。")

            # 创建索引以优化查询性能
            print("[INFO] 正在为 'Entity' 节点的 'name' 属性创建索引...")
            try:
                session.run("CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)")
                print("[INFO] 索引创建成功。")
            except Exception as e:
                print(f"[WARN] 创建索引时出错: {e}")

    finally:
        driver.close()
        print("[INFO] 图谱构建流程结束。")


if __name__ == '__main__':
    # 从环境变量读取配置
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    build_graph(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        neo4j_database=NEO4J_DATABASE
    )
