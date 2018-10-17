#!/usr/bin/env python3
# coding:utf-8
from pylab import *
with open('weibo_train_data.txt','r') as f:
    all_message=f.readlines()
user_id=[]
weibo_id=[]
post_time=[]
f_count=[]
c_count=[]
l_count=[]
content=[]
for i in all_message:
    #j=i.split('\t')
    user_id.append(i.split('\t')[0])
    #weibo_id.append(j[1])
    #post_time.append(j[2])
    #f_count.append(j[3])
    #c_count.append(j[4])
    #l_count.append(j[5])
    #content.append(j[6])

print("微博数目：",len(user_id))
print("博主数目：",len(set(user_id)))

x=set(user_id)
y=[]
z={}

for i in user_id:
    if(z.setdefault(i) is None):
        z[i]=1
    else:
        z[i]+=1


print('ok')

plot(list(range(len(z))),list(z.values()))
show()
