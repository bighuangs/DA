import os

# 防止系统代理（Clash 等）拦截本地 Docker 服务请求，需在所有 http 客户端初始化前设置
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")

import uvicorn
from app.bootstrap.api import create_app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
