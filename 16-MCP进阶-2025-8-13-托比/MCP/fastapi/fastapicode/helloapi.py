# 第一步：导入FastAPI
from fastapi import FastAPI

# 第二步：创建FastAPI实例，相当于初始化Web服务
app = FastAPI()

# 第三步：定义路由 - 访问/hello路径时执行的功能
@app.get("/hello")
async def say_hello():
    # 返回的字典会自动转换为JSON格式
    return {"message": "Hello FastAPI!"}