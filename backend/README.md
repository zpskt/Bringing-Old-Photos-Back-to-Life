# 配置后端Django服务

参考链接: 
1. https://docs.djangoproject.com/zh-hans/4.2/
## 环境
python 3.9.x
Django 4.2
## 安装
```shell
pip install Django
```
### 验证方法1 
```shell
python
```
```python
import django
django.get_version()
```
返回正确的版本号，说明安装成功

### 验证方法2
```shell
django-admin
```
正常返回信息说明安装成功
## 创建Django项目
```shell
django-admin startproject ${your_project}
```
执行完此命令后，会在当前目录下生成一个名为your_project的文件夹.

### 目录结构
```shell
cd your_project
tree /f
```
目录结构如下图所示
```
D:.
│  manage.py
│
└─repaired_old_photo
        asgi.py
        settings.py
        urls.py
        wsgi.py
        __init__.py
```
目录结构说明：
1. HelloWorld: 项目的容器。
2. manage.py: 一个实用的命令行工具，可让你以各种方式与该 Django 项目进行交互。
3. HelloWorld/__init__.py: 一个空文件，告诉 Python 该目录是一个 Python 包。
4. HelloWorld/asgi.py: 一个 ASGI 兼容的 Web 服务器的入口，以便运行你的项目。
5. HelloWorld/settings.py: 该 Django 项目的设置/配置。
6. HelloWorld/urls.py: 该 Django 项目的 URL 声明; 一份由 Django 驱动的网站"目录"。
7. HelloWorld/wsgi.py: 一个 WSGI 兼容的 Web 服务器的入口，以便运行你的项目。

## 启动服务
```shell
python ./manage.py runserver 0.0.0.0:8000
```

说明：0.0.0.0 让其它电脑可连接到开发服务器，8000 为端口号。如果不说明，那么端口号默认为 8000。

现在访问[http://localhost:8000/] 
即可访问到Django服务了。
### 管理页
 访问 http://localhost:8000/admin 进入登录页
 默认账号密码：admin/admin
 
## 创建应用
创建应用，即创建一个可独立运行的模块，比如一个博客应用，一个商城应用，一个论坛应用，一个问答应用，一个问答应用等等。
这里演示创建一个投票应用
```shell
python ./manage.py startapp polls 
```
此时的目录结构如下：
```
D:.
│  db.sqlite3
│  manage.py
│  
├─polls
│  │  admin.py
│  │  apps.py
│  │  models.py
│  │  tests.py
│  │  views.py
│  │  __init__.py
│  │  
│  └─migrations
│          __init__.py
│
└─repaired_old_photo
    │  asgi.py
    │  settings.py
    │  urls.py
    │  wsgi.py
    │  __init__.py
    │
    └─__pycache__
            settings.cpython-37.pyc
            urls.cpython-37.pyc
            wsgi.cpython-37.pyc
            __init__.cpython-37.pyc
```
### 构建视图
