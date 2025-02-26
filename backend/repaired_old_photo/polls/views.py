from django.http import HttpResponse, Http404
from django.shortcuts import render, get_object_or_404
from django.template import loader

from .models import Question


# 定义一个名为 index 的视图函数，用于处理根路径的请求
def index(request):
    latest_question_list = Question.objects.order_by("-pub_date")[:5]
    context = {
        "latest_question_list": latest_question_list,
    }
    return render(request, "polls/index.html", context)

# 定义一个名为 detail 的视图函数，用于处理特定问题详情页的请求
# 参数 request 是 Django 传入的 HTTP 请求对象，question_id 是问题的唯一标识符
def detail(request, question_id):
    # 返回一个 HttpResponse 对象，显示正在查看的问题的 ID
    question = get_object_or_404(Question, pk=question_id)
    return render(request, "polls/detail.html", {"question": question})

# 定义一个名为 results 的视图函数，用于处理特定问题结果页的请求
# 参数 request 是 Django 传入的 HTTP 请求对象，question_id 是问题的唯一标识符
def results(request, question_id):
    # 定义一个字符串变量，用于存储结果页的提示信息
    response = "You're looking at the results of question %s."
    # 返回一个 HttpResponse 对象，显示正在查看的问题的结果
    return HttpResponse(response % question_id)

# 定义一个名为 vote 的视图函数，用于处理特定问题投票页的请求
# 参数 request 是 Django 传入的 HTTP 请求对象，question_id 是问题的唯一标识符
def vote(request, question_id):
    # 返回一个 HttpResponse 对象，显示正在对某个问题进行投票
    return HttpResponse("You're voting on question %s." % question_id)
