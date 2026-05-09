import json
import requests
import sys
import time

def test_pipeline_with_async_task():
    # 1. 基础配置
    base_url = "http://127.0.0.1:8613/vectorization"
    run_url = f"{base_url}/run"
    
    # 2. 准备请求参数
    payload = {
        "user_id": "admin_user",
        "kb_id": "hero_kb_001",
        "docment_id": "hero_doc_001",
        "json_path": "/opt/Workspace/CRX/NextGraph/《千面英雄》-约瑟夫-坎贝尔_content_list_enhanced_textified.json",
        "max_concurrency": 14,
        "max_chunk_chars": 1500
    }

    print(f"--- 启动异步任务测试 ---")
    print(f"目标文件: {payload['json_path']}")
    
    try:
        # 第一步：提交任务
        print(f"1. 正在提交任务到: {run_url}...")
        resp = requests.post(run_url, json=payload)
        if resp.status_code != 200:
            print(f"提交失败: {resp.text}")
            return
        
        task_info = resp.json()
        task_id = task_info.get("task_id")
        print(f"✅ 任务已创建，ID: {task_id}")

        # 第二步：连接流式进度接口
        stream_url = f"{base_url}/stream/{task_id}"
        print(f"2. 正在连接进度流: {stream_url}...\n")
        
        # 增加一点点延迟确保后台任务已初始化
        time.sleep(0.5)
        
        # 使用 GET 请求访问流式接口
        response = requests.get(stream_url, stream=True, timeout=600)
        
        # 逐行处理 SSE (Server-Sent Events) 数据
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    # 解析 JSON 数据
                    try:
                        data = json.loads(decoded_line[6:])
                        step = data.get("step", "")
                        msg = data.get("msg", "")
                        progress = data.get("progress", 0)
                        
                        # 格式化输出进度
                        print(f"[{step:<12}] ({progress:>4.0%}) {msg}")
                        
                        if step == "completed":
                            summary = data.get("summary", {})
                            print(f"\n✅ 处理成功！最终统计: {summary}")
                            break
                            
                        if step == "error":
                            print(f"\n❌ 任务运行失败: {msg}")
                            break
                            
                    except json.JSONDecodeError:
                        print(f"无法解析数据: {decoded_line}")

    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务器，请检查后端是否已启动 (python main.py)")
    except Exception as e:
        print(f"测试异常: {str(e)}")

if __name__ == "__main__":
    test_pipeline_with_async_task()
