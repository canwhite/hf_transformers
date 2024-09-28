'''
这里主要用于测试我的工具包
'''
from jssyntax import List,Map,Set,Singleton


mylist = List([])
mylist.push(1)
mylist.push(2)
mylist.forEach(lambda x: print(x))

@Singleton
class TestSingleton:
    def __init__(self,name):
        print("----1")
        self.name = name
    
    def getName(self): #注意不要忘了self 
        return self.name


test =  TestSingleton("test")
print(test.getName())











