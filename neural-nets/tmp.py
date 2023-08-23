x = 100


def myfn():
    global x
    x += 5


myfn()
print(x)
