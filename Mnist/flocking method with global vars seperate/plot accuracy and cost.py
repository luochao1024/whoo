import numpy as np
import csv
import matplotlib.pyplot as plt



with open ("flocking_noncenter_1.000_3.000_worker_2_r0.1.txt", 'r') as csvfile:
    lines1 = csv.reader(csvfile, delimiter=' ')
    line_list1 = [e for e in lines1]
    num_lines1 = len(line_list1)
    cost_flocking = np.array([0.0]*num_lines1)
    accuracy_flocking = np.array([0.0]*num_lines1)
    for i in range(num_lines1):
        cost_flocking[i]= line_list1[i][0]
        accuracy_flocking[i] = line_list1[i][1]

with open ("flocking_noncenter_0.000_0.000_worker_2_r0.1.txt", 'r') as csvfile:
    lines2 = csv.reader(csvfile, delimiter=' ')
    line_list2 = [e for e in lines2]
    num_lines2 = len(line_list2)
    cost_nonflocking = np.array([0.0]*num_lines2)
    accuracy_nonflocking = np.array([0.0]*num_lines2)
    for i in range(num_lines2):
        cost_nonflocking[i]= line_list2[i][0]
        accuracy_nonflocking[i] = line_list2[i][1]
        
#print(cost_flocking)


x1 = np.linspace(0, 48,num_lines1 )
x2 = np.linspace(0, 48, num_lines2)
print(x1)

#plt.ylim((0, 1))
#plt.xlim((0, 48))
#plt.xlabel('time(seconds)')
#plt.ylabel('accuracy')
#plt.plot(x1, accuracy_flocking)
#plt.plot(x2, accuracy_nonflocking)
#plt.legend(['flocking', 'nonflocking'])
#plt.show()

plt.ylim((0, 3))
plt.xlim((0, 48))
plt.xlabel('time(seconds)')
plt.ylabel('cost')
plt.plot(x1, cost_flocking)
plt.plot(x2, cost_nonflocking)
plt.legend(['flocking', 'nonflocking'])
plt.show()