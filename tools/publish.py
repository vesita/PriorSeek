import os
import sys
from modelscope.hub.api import HubApi

def publish_artifacts(namespace, repo_name):
    # 读取 token
    token_path = os.path.expanduser("~/coding/my/keys/modelscope.token")
    if not os.path.exists(token_path):
        print("Token file not found")
        sys.exit(1)
        
    with open(token_path, 'r') as f:
        token = f.read().split('=')[1].strip()
        
    api = HubApi()
    api.login(token)
    
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifact_dir = os.path.join(root_dir, "data/scifact/artifacts")
    
    # 上传逻辑：遍历 artifacts 目录
    for artifact in os.listdir(artifact_dir):
        path = os.path.join(artifact_dir, artifact)
        if os.path.isdir(path):
            print(f"Uploading artifact: {artifact}")
            # 这里简化演示，实际使用 api.upload_file/folder
            # api.push_model(model_id=f"{namespace}/{repo_name}", folder_path=path)
            pass
    print("Publishing simulation complete.")

if __name__ == "__main__":
    publish_artifacts("Reasone", "PriorSeek")
