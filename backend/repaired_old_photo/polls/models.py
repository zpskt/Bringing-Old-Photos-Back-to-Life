from django.utils import timezone
import datetime
from django.db import models


# 定义一个名为 Question 的 Django 模型类，继承自 models.Model
class Question(models.Model):
    def __str__(self):
        return self.question_text
    # 定义一个 CharField 字段，用于存储问题的文本内容，最大长度为 200 个字符
    question_text = models.CharField(max_length=200)
    # 定义一个 DateTimeField 字段，用于存储问题的发布日期，人类可读的名称为 "date published"
    pub_date = models.DateTimeField("date published")
    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)


# 定义一个名为 Choice 的 Django 模型类，继承自 models.Model
class Choice(models.Model):
    # ...
    def __str__(self):
        return self.choice_text
    # 定义一个外键字段，关联到 Question 模型，当关联的 Question 对象被删除时，此 Choice 对象也会被级联删除
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    # 定义一个 CharField 字段，用于存储选项的文本内容，最大长度为 200 个字符
    choice_text = models.CharField(max_length=200)
    # 定义一个 IntegerField 字段，用于存储该选项的投票数，默认值为 0
    votes = models.IntegerField(default=0)
