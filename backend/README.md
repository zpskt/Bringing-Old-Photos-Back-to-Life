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
###  新建数据库
#### 服务器安装mysql
实验环境为centos7
官方地址： https://dev.mysql.com/doc/refman/8.0/en/linux-installation.html
我这里安装的5.7版本 
```shell
yum repolist all | grep mysql
sudo yum install mysql-community-server
```
如果显示GPG错误，找到[mysql5.7-community] 修改：gpgcheck参数为0，即不校验gpg
```shell
vi /etc/yum.repos.d/mysql-community.repo
```
安装完成后，启动mysql服务
```shell
sudo systemctl start mysqld
sudo systemctl enable mysqld
```
获取临时密码
```shell
sudo grep 'temporary password' /var/log/mysqld.log
```
拿着你的临时密码就可以进行初始化设置了
3,<Ld?!D_SsZ
配置mysql
```shell
sudo mysql_secure_installation
```
设置mysql端口号
```shell
sudo vi /etc/my.cnf
```
[mysqld]
port = 3306
设置防火墙允许访问
```shell
sudo iptables -L -n
sudo iptables -A INPUT -p tcp --dport 3306 -j ACCEPT
sudo netstat -tuln | grep 3306
```
在服务器本身设置root账户可以远程访问
```shell
mysql -u root -p
```
```mysql
use mysql;
select host from user where user='root';
update user set host = '%' where user ='root';
flush privileges;
select host from user where user='root';
```

```shell
sudo systemctl restart mysqld
```
好了，现在很尴尬，Django支持json，但是mysql5.7不支持json，所以需要安装mysql8.0
由于mysql8.0
所以需要进行单独的属性配置IDE或者是链接参数加上 allowPublicKeyRetrieval
allowPublicKeyRetrieval=true



#### 安装依赖
安装python中的mysqlclient库
```shell
conda install mysqlclient
```
如果是比较老版本的python
那么还需要再安装一个
```shell
pip install MySQL-python
```
系统以及数据库相关文件都在 settings.py文件中

执行命令新建表格
```shell
python manage.py migrate
```

### 常用的命令
1. python manage.py runserver 启动服务
2. python manage.py startapp polls 创建应用
3. python manage.py migrate 新建数据库
4. python manage.py makemigrations polls 生成迁移文件
5. python manage.py sqlmigrate polls 0001 查看迁移文件
6. python manage.py check 检查项目的问题
7. python manage.py shell 进入python shell
8. python manage.py test polls 测试应用
9. python manage.py createsuperuser 创建超级用户
10. python manage.py dbshell 进入数据库
11. python manage.py flush 清空数据库
12. python manage.py inspectdb > polls/models.py 从数据库中生成模型
13. python manage.py sqlall polls 生成创建所有应用的SQL语句
14. python manage.py validate 验证项目的完整性
15. python manage.py showmigrations polls 显示应用的迁移历史
16. python manage.py migrate polls 0001 应用迁移
17. python manage.py migrate polls 0001-0002 应用迁移
18. python manage.py migrate polls --fake 应用迁移
19. python manage.py migrate polls --fake-initial 应用迁移
20. python manage.py migrate polls --unapply 应用迁移
21. python manage.py migrate polls --list 应用迁移
22. python manage.py migrate polls --plan 应用迁移
23. python manage.py migrate polls --database=otherdb 应用迁移
24. python manage.py migrate polls --database=otherdb --fake 应用迁移

### 开发流程
1. 创建应用
2. 建立路由关系
3. settings.py中配置路由关系
4. 编辑 models.py 文件，改变模型
5. 运行 python manage.py makemigrations 为模型的改变生成迁移文件。 
6. 运行 python manage.py migrate 来应用数据库迁移。