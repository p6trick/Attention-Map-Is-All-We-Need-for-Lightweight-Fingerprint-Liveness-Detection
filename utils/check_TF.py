def check_TF(dset):
    for d in ['x_target', 'x_label']:
        t_num = 0
        f_num = 0
        for i in dset[d]:
            if type(i) == list:
                i = i.argmax()
            if i == 1:
                t_num += 1
            else:
                f_num += 1
        dset[f'{d}_t_num'] = t_num
        dset[f'{d}_f_num'] = f_num
        print(f'# of {d} t_num: {dset[f"{d}_t_num"]}')
        print(f'# of {d} f_num: {dset[f"{d}_f_num"]}')
    return dset