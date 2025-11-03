import requests
import json
from datetime import datetime

def check_elasticsearch():
    base_url = "http://localhost:9200"
    
    print("=== Elasticsearch 健康检查 ===")
    try:
        # 1. 检查集群健康状态
        health_response = requests.get(f"{base_url}/_cluster/health")
        health_data = health_response.json()
        print(f"集群状态: {health_data.get('status', 'unknown')}")
        print(f"节点数量: {health_data.get('number_of_nodes', 0)}")
        print(f"数据节点: {health_data.get('number_of_data_nodes', 0)}")
        print()
        
        # 2. 查看所有索引
        indices_response = requests.get(f"{base_url}/_cat/indices?format=json")
        indices_data = indices_response.json()
        print("=== 索引列表 ===")
        for index in indices_data:
            print(f"索引: {index['index']}, 文档数: {index['docs.count']}, 大小: {index['store.size']}")
        print()
        
        # 3. 搜索最新的日志
        search_query = {
            "query": {"match_all": {}},
            "size": 5,
            "sort": [{"@timestamp": {"order": "desc"}}]
        }
        
        search_response = requests.post(
            f"{base_url}/_search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(search_query)
        )
        
        if search_response.status_code == 200:
            search_data = search_response.json()
            hits = search_data.get('hits', {}).get('hits', [])
            
            print("=== 最新的5条日志 ===")
            if hits:
                for i, hit in enumerate(hits, 1):
                    source = hit['_source']
                    timestamp = source.get('@timestamp', 'N/A')
                    level = source.get('level', 'N/A')
                    message = source.get('message', 'N/A')
                    module = source.get('module', 'N/A')
                    
                    print(f"{i}. [{timestamp}] {level} - {module}: {message}")
            else:
                print("没有找到日志数据")
        else:
            print(f"搜索失败: {search_response.status_code} - {search_response.text}")
            
        # 4. 专门搜索ELK日志索引
        elk_search_response = requests.post(
            f"{base_url}/elk-logs-*/_search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(search_query)
        )
        
        if elk_search_response.status_code == 200:
            elk_data = elk_search_response.json()
            elk_hits = elk_data.get('hits', {}).get('hits', [])
            total_hits = elk_data.get('hits', {}).get('total', {})
            
            print(f"\n=== ELK日志索引统计 ===")
            if isinstance(total_hits, dict):
                print(f"总日志数量: {total_hits.get('value', 0)}")
            else:
                print(f"总日志数量: {total_hits}")
                
            if elk_hits:
                print("最新ELK日志:")
                for i, hit in enumerate(elk_hits[:3], 1):
                    source = hit['_source']
                    timestamp = source.get('@timestamp', 'N/A')
                    level = source.get('level', 'N/A')
                    message = source.get('message', 'N/A')[:100]  # 截取前100字符
                    print(f"{i}. [{timestamp}] {level}: {message}...")
        else:
            print(f"ELK索引搜索失败: {elk_search_response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Elasticsearch (http://localhost:9200)")
    except Exception as e:
        print(f"❌ 检查过程中出现错误: {e}")

if __name__ == "__main__":
    check_elasticsearch()