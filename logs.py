import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

plt.rcParams['font.size'] = 12  # 设置全局字体大小
plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['font.style'] = 'italic'
plt.rcParams['figure.dpi'] = 300


def read_file(file_path):
    pre = []
    re = []
    f1 = []
    with open(file_path, 'r') as file:
    # 逐行读取文件内容
        for line in file:
            if("Evaluation" in line):
                pre.append(float(line[22:29]))
                re.append(float(line[39:46]))
                f1.append(float(line[52:]))
    return pre, re, f1

def log_pic(list1, list2, name):
    plt.figure(figsize=(6, 4))
    plt.plot(list1, color='red', label='my')
    plt.plot(list2, color='blue', label = 'base')
    plt.title(name)
    plt.legend()
    plt.savefig('logs/'+name+'.png')
    plt.clf()

def log_pic_f1(list1, list2, list3, name):
    plt.figure(figsize=(6, 4))
    plt.xlabel('Trainning Steps')
    plt.ylabel('F1 Score')
    plt.plot(list1, color='red', label='my')
    plt.plot(list2, color='blue', label = 'base')
    # plt.plot(list3, color='green', label='my_win')
    # plt.plot(list4, color='black', label = 'base_win')
    plt.title(name)
    plt.legend()
    plt.savefig('logs/'+name+'.png')
    plt.clf()

# def log_pic_f1(list1, list2, list3, list4, name):
#     # x = np.arange(len(data))
#     plt.figure(figsize=(6, 4))
#     plt.plot(list1, color='red', label='Base')
#     plt.plot(list2, color='blue', label = 'SIA')
#     plt.plot(list3, color='green', label = 'Independent Branches')
#     plt.plot(list4, color='purple', label = 'Complete Model')
#     # plt.plot(list3, color='green', label='my_win')
#     # plt.plot(list4, color='black', label = 'base_win')
#     plt.title(name)
#     plt.legend()
#     plt.savefig('logs/'+name+'.png')
#     plt.clf()

def log_pic_f1(list1, list2, list3, list4, name):
    x = np.arange(len(list1))
    x2 = np.arange(len(list4))
    bspline1 = UnivariateSpline(x, list1, s=0.07)
    bspline2 = UnivariateSpline(x, list2, s=0.07)
    bspline3 = UnivariateSpline(x, list3, s=0.07)
    bspline4 = UnivariateSpline(x2, list4, s=0.07)
    xnew1 = np.linspace(0, len(list1)-1, num=3000)
    xnew2 = np.linspace(0, len(list1)-1, num=3000)
    xnew3 = np.linspace(0, len(list1)-1, num=3000)
    xnew4 = np.linspace(0, len(list1)-1, num=3000)
    y_smooth1 = bspline1(xnew1)
    y_smooth2 = bspline2(xnew2)
    y_smooth3 = bspline3(xnew3)
    y_smooth4 = bspline4(xnew4)
    print(y_smooth1)
    for i in range(len(y_smooth1)):
        if y_smooth1[i] < 0:
            y_smooth1[i] = 0
    for i in range(len(y_smooth2)):
        if y_smooth2[i] < 0:
            y_smooth2[i] = 0
    for i in range(len(y_smooth3)):
        if y_smooth3[i] < 0:
            y_smooth3[i] = 0
    for i in range(len(y_smooth4)):
        if y_smooth4[i] < 0:
            y_smooth4[i] = 0


    plt.figure(figsize=(6, 5))
    plt.xlabel('Trainning Steps')
    plt.ylabel('F1 Score')
    plt.plot(y_smooth1, color='red', label='Base')
    plt.plot(y_smooth2, color='blue', label = 'SIA')
    plt.plot(y_smooth3, color='green', label = 'IB')
    plt.plot(y_smooth4, color='purple', label = 'Complete Model')
    # plt.plot(list3, color='green', label='my_win')
    # plt.plot(list4, color='black', label = 'base_win')
    # plt.title(name)
    plt.legend()
    plt.savefig('logs/'+name+'.png')
    plt.clf()

pr_base, re_base, f1_base = read_file('/2014110093/transformers_tasks/UIE-SIAIB/tr_log_base.txt')
pr_bra, re_bra, f1_bra = read_file('/2014110093/transformers_tasks/UIE-SIAIB/tr_log_onlybranch-.txt')
# pr1, re1, f1 = read_file('/2014110093/transformers_tasks/UIE-SIAIB/tr_log_crossbranch.txt')
# pr2, re2, f2 = read_file('/2014110093/transformers_tasks/UIE-SIAIB/tr_log_all--.txt')
pr_sia, re_sia, f1_sia = read_file('/2014110093/transformers_tasks/UIE-SIAIB/tr_log_onlysegment-.txt')
pr_all, re_all, f1_all = read_file('/2014110093/transformers_tasks/UIE-SIAIB/tr_log_all--2.txt')


# list_f1_a = []
# win = 0
# lose = 0
# # list_f1_b = []
# # f1 = f1[100:]

# for f1_a, f1_b in zip(f1, f2):
#     if f1_a >= 0.5 and f1_b > 0.5:
#         if f1_a >= f1_b:
#             list_f1_a.append(1.2)
#             win += 1
#         # list_f1_b.append(0)
#         else:
#             list_f1_a.append(1)
#             lose += 1
#         # list_f1_b.append(1)


# log_pic(pr1, pr2, 'pre2')
# log_pic(re1, re2,  're2')
log_pic_f1(f1_base, f1_sia, f1_bra, f1_all, 'F1 Comparison')
log_pic_f1(pr_base, pr_sia, pr_bra, pr_all, 'PR Comparison')
log_pic_f1(re_base, re_sia, re_bra, re_all, 'RE Comparison')
# print('win', win, 'lose', lose)


