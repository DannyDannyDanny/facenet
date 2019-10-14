import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

mat_pat = "../clustering_results/dist_matrix_2634A.npy"
import os

os.listdir("../")

fnames_pat = "../clustering_results/img_fnames_2634A.npy"
mat = np.load(mat_pat)
img_fnames = np.load(fnames_pat)
print(mat.shape)
print(len(img_fnames))

sns.heatmap(mat)
plt.show()

# %%
wonder_dict = {}

for i in range(mat.shape[0])[-10:]:
    for j in range(mat.shape[0])[-10:]:
        print(i, j, mat[i, j])
        img_fname_i = img_fnames[i]
        img_fname_j = img_fnames[j]

print("i\tj\tsp\tac\t...")
for i in range(mat.shape[0]):
    for j in range(mat.shape[0]):
        ii = img_fnames[i]
        ij = img_fnames[j]
        same_person = ii.split("_")[0] == ij.split("_")[0]

        if same_person:
            same_original_img = ii.split("_")[1] == ij.split("_")[1]
        else:
            same_original_img = False

        ii_A = "_A.png" in ii
        ij_A = "_A.png" in ij
        if ii_A and ij_A:
            anon_combi = "aa"
        if not ii_A and not ij_A:
            anon_combi = "oo"
        else:
            anon_combi = "ao"

        sp = "s" if same_person else "d"
        spac = sp + "_" + anon_combi
        if not same_original_img:
            # print(f"{i}\t{j}\t{same_person}\t{same_original_img}\t{anon_combi}", ii, ij)
            # print(f"{i}\t{j}\t{sp}\t{anon_combi}\t{mat[i][j]}")
            # print(f"{i}\t{j}\t{sp}\t{anon_combi}\t{mat[i][j]}")
            if spac not in wonder_dict.keys():
                print("adding", spac, "to wonder dict")
                wonder_dict[spac] = []

            wonder_dict[spac].append(mat[i][j])

# %%
wonder_dict.keys()

# %%
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

wdn = {k: normalize([v])[0] for k, v in wonder_dict.items()}
wdn2 = {k: [p / len(v) for p in v] for k, v in wonder_dict.items()}
# %%
axes = plt.gca()


# %%
for k, v in wdn2.items():
    v = np.ones_like(v) / float(len(v))
    print(k, len(v), v[0:3])


# %%
v = np.ones_like(v) / float(len(v))
plt.hist(myarray, weights=weights)


# %%
# axes.set_xlim([-1e-3, 0.02])
for k, v in wdn2.items():
    print(k, len(v))
    v_norm = np.ones_like(v) / float(len(v))
    # plt.hist(([v]), label=k, alpha=0.5)
    plt.hist(v, weights=v_norm, label=k)
plt.legend()
plt.show()


# %%

wonder_dict.keys()
# %%

img_fnames.shape

sns_plot = sns.heatmap(mat, cbar=False)
# %%

i
