import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import sys

crm_dist = np.loadtxt(sys.argv[1], delimiter='\t', usecols=(7))
ncrm_dist = np.loadtxt(sys.argv[2], delimiter='\t', usecols=(6))

sns.kdeplot(crm_dist, label="CRM Distances")
sns.kdeplot(ncrm_dist, label="NCRM Distances")
plt.xlabel("Distance from N")
plt.legend()
plt.savefig("crm_ncrm_distance_from_n.png", dpi=600)