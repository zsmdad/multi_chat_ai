# 使用多阶段构建来减小最终镜像大小
# 第一阶段：使用Python官方镜像安装依赖
FROM python:3.12-slim AS builder

# 设置工作目录
WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装依赖到指定目录
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 第二阶段：创建最终镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 从第一阶段复制已安装的依赖
COPY --from=builder /install /usr/local

# 复制应用代码
COPY app.py .
COPY startup.sh .

# 赋予启动脚本执行权限
RUN chmod +x startup.sh

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# 暴露端口
EXPOSE 7860

# 启动应用
CMD ["./startup.sh"]