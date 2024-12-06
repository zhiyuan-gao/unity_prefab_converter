# vertex = [[ 4.049 ,      0.             ],
# [ 4.049  ,     2.62125381       ],
# [ 7.63713789 , 0.               ],
# [ 7.63713789,  2.13022733      ],
# [10.123      , 2.62125381      ],
# [ 9.63406026  ,2.13022733      ],
# [10.123       ,0.            ],
# [ 9.63406026  ,0.                ]]


import matplotlib.pyplot as plt

# 点数据
# vertices = [[0.     ,    0.        ],
#  [2.62125381 ,0.        ],
#  [0.        , 1.28692796],
#  [2.07534075 ,1.28692796],
#  [2.62125381 ,8.098     ],
#  [2.07534075 ,3.19718029],
#  [0.       ,  8.098     ],
#  [0.      ,   3.19718029]]



vertices = [[0.        , 0.        ],
 [0.       ,  8.098     ],
 [0.    ,     1.28692796],
 [0.     ,    3.19718029],
 [2.62125381 ,8.098     ],
 [2.07534075 ,3.19718029],
 [2.62125381, 0.        ],
 [2.07534075, 1.28692796]]




# 转置数据以方便绘图
x, y = zip(*vertices)

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制点
plt.scatter(x, y, color='blue', label='Vertices')

# 标注点索引
for i, (xi, yi) in enumerate(vertices):
    plt.text(xi, yi, str(i), fontsize=10, color='red', ha='center', va='bottom')

# 设置坐标轴和标题
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Vertex Points with Indices")
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

# 设置图例
plt.legend()

# 显示图形
plt.show()
