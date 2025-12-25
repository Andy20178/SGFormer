P_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
O_all = [i for i in range(160)]
def get_task_config(task_id, mode):
    if mode == 'S1':
        predicates = P_all[:10 * (task_id + 1)]  # 前10，前20...
        objects = O_all   # 全部 object 都一样
    # elif mode == 'S2':
    #     predicates = P_task[task_id]
    #     objects = O_task[task_id]
    # elif mode == 'S3':
    #     predicates = P_all  # 全部 predicates
    #     objects = O_all[:30 * (task_id + 1)]  # objects 递增
    
    return objects, predicates
