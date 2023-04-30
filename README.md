# HOKAI

AI for Honor of Kings

王者荣耀游戏AI

```
python HOKAI.py

```

当前使用mumu模拟器 如果使用其他模拟器 需要更改adb连接设置

分辨率为1600*900

Data

0 - no hero
1 - only self hero
2 - only enemy hero
3 - both self and enemy hero


一些记录
Inception v1和v2 用的是四分类
分类标准为 一个patch里包含的单位
0 - 无
1 - 仅我方英雄
2 - 仅敌方英雄
3 - 双方英雄
但是3类别特别少 0类别特别多 导致样本不均衡 结果就是网络会把所有输入分到同一个类别
解决方案为 取消类别3 让三个类别的训练样本尽可能均衡

敌我英雄是否阵亡 - 直接检查对应像素的颜色阈值 最简单 最高效

注意 mumu模拟器是管理员权限运行 所以vscode或者运行代码的cmd也要以管理员权限启动 否则无法模拟输入

移动：WASD

攻击：J

记录玩家移动需要在模拟器上在王者移动轮盘的位置放一个WASD轮盘

AI移动是通过mumu模拟器的鼠标移动功能实现 需要在王者移动轮盘的位置放一个鼠标轮盘
