import matplotlib.pyplot as plt
from main import *
blr, rto, sed, sq_lst, edd, edd_lst, emd, gll, gl_list, gla_stress = getRealtime(2)

rto = np.array(rto)
rto = np.array(rto*100,dtype=int)
sq_lst = np.array(sq_lst)
sq_lst = np.array(sq_lst*100, dtype=int)
gl_list =np.array(gl_list, dtype=int)
gla_stress = np.array(gla_stress, dtype=int)
edd_lst = np.array(edd_lst, dtype=int)
print(rto)
fig, ax = plt.subplots()
ax.plot(rto, label='EAR and Squint Duration', linewidth=2, color='blue')
ax.plot( rto[sq_lst], color='orange', linewidth=2)
# Add a title and axis labels
plt.xlabel('Time Frame')
plt.ylabel('Aspect Ratio')
# Show the plot
plt.show()

fig1, ax1 = plt.subplots()
ax1.plot(gl_list, label='Glabellar Length', linewidth=2, color='blue')
ax1.plot( gl_list[gla_stress], color='orange', linewidth=2)
# Add a title and axis labels
plt.xlabel('Time Frame')
plt.ylabel('Glabellar Length')
# Show the plot
plt.show()


fig2, ay = plt.subplots()
ay.plot(edd_lst, label='Eye Device Distance', linewidth=2, color='green')
plt.xlabel('Time Frame')
plt.ylabel('Eye Device Distance')
# Show the plot
plt.show()


