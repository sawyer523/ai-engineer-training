def my_decorator(func):
    def wrapper():
        print("调用函数前执行一些操作...")
        func()
        print("调用函数后执行一些操作...")
    return wrapper

@my_decorator  # 等价于 say_hello = my_decorator(say_hello)
def say_hello():
    print("Hello, World!")

say_hello()


print(say_hello.__name__)