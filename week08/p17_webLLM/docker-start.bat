@echo off
chcp 65001 >nul

echo 🚀 启动WebLLM Docker服务...

REM 检查.env文件是否存在
if not exist .env (
    echo ⚠️  .env文件不存在，复制.env.docker为.env
    copy .env.docker .env >nul
    echo 📝 请编辑.env文件，填入正确的API密钥等配置
    echo 💡 特别注意设置DASHSCOPE_API_KEY
)

REM 停止现有容器
echo 🛑 停止现有容器...
docker-compose down

REM 构建并启动服务
echo 🔨 构建并启动服务...
docker-compose up --build -d

REM 等待服务启动
echo ⏳ 等待服务启动...
timeout /t 10 /nobreak >nul

REM 检查服务状态
echo 📊 检查服务状态...
docker-compose ps

echo.
echo ✅ 服务启动完成！
echo 🌐 FastAPI服务: http://localhost:8000
echo 🎨 Gradio界面: http://localhost:7860
echo 📊 健康检查: http://localhost:8000/health
echo.
echo 📝 查看日志: docker-compose logs -f
echo 🛑 停止服务: docker-compose down

pause