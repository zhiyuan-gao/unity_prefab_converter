def polygon_wall_to_simple_wall(wall, holes):
    # 假设wall['polygon']是一个包含点的列表，并且点是具有x和y属性的字典
    polygons = sorted(wall['polygon'], key=lambda p: p['y'])
    max_y = max(p['y'] for p in wall['polygon'])
    
    # 获取hole
    hole = holes.get(wall['id'], None)
    
    # 获取第一个点p0
    p0 = polygons[0]
    
    # 返回一个字典而不是Wall对象
    return {
        'id': wall['id'],
        'p0': polygons[0],
        'p1': polygons[1],
        'height': max_y - p0['y'],
        'material': wall['material'],
        'empty': wall['empty'],
        'roomId': wall['roomId'],
        'thickness': wall['thickness'],
        'hole': hole,
        'layer': wall['layer']
    }

# 示例输入，假设wall是一个字典，holes是一个字典
wall = {
    'id': 'wall1',
    'polygon': [{'x': 0, 'y': 0}, {'x': 1, 'y': 5}, {'x': 2, 'y': 3}],
    'material': 'brick',
    'empty': False,
    'roomId': 'roomA',
    'thickness': 0.3,
    'layer': 1
}

holes = {
    'wall1': {'id': 'hole1'}
}

# 调用函数
simple_wall = polygon_wall_to_simple_wall(wall, holes)
print(simple_wall)
