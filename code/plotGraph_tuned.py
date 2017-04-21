import matplotlib.pyplot as plt
import numpy as nump

# precision_untuned_rf = [82.12890625, 25.0, 38.3040970143, 35.202004261, 40.9466818642, 77.35, 28.2827800334, 36.8259803922, 79.5454545455, 7.99419182878, 22.1161198195, 9.55334987593, 48.5763501546, 36.5603028664, 34.6581196581, 29.347826087, 14.7997351223]
# precision_untuned_ct = [82.12890625, 25.0, 43.7504735925, 35.8100661881, 39.4143755181, 69.9789915966, 37.6046738072, 47.8174603175, 92.7272727273, 22.0097845515, 24.3339248665, 34.8507869721, 56.3590374179, 36.5603028664, 36.5555555556, 28.4496124031, 16.17081202]
# precision_tuned_rf = [82.12890625, 25.0, 41.3869937582, 23.9762187872, 58.3274231678, 74.5454545455, 36.2115732369, 39.2455732369, 92.9998878693, 18.56332523, 24.56772993, 37.7798878693, 51.1657653778, 36.5603028664, 32.56332523, 37.5037481259, 15.4809164315]
# precision_tuned_ct = [87.1691176471, 25.0, 38.1051596288, 23.9762187872, 40.0897565791, 74.8275862069, 37.2498979175, 51.0416666667, 92.7272727273, 25.58944424, 24.600808795, 39.6451673357, 51.1657653778, 37.430786268, 36.9049700891, 30.5858653685, 42.6623992413]
#
# precision_untuned_rf = nump.asarray(precision_untuned_rf).astype(float)
# precision_tuned_rf = nump.asarray(precision_tuned_rf).astype(float)
# precision_untuned_ct = nump.asarray(precision_untuned_ct).astype(float)
# precision_tuned_ct = nump.asarray(precision_tuned_ct).astype(float)
#
# delta_precision_rf = precision_tuned_rf - precision_untuned_rf
# print delta_precision_rf
# delta_precision_ct = precision_tuned_ct - precision_untuned_ct
# print delta_precision_ct
#
# dataset = []
#
# for i in range(1,18):
#     dataset.append(i)
#
# plt.title("Delta of Precision between tuned and untuned Random Forest, CART")
# plt.plot(dataset, delta_precision_rf, color = 'g', label='Random Forest')
# plt.plot(dataset, delta_precision_ct, color = 'y', label='CART')
# plt.legend()
# plt.show()

f_score_untuned_rf = [86.1680327869, 33.3333333333, 43.5134912564, 39.6645928784, 44.8879497393, 64.6412411118, 35.7882623705, 38.1048387097, 70.7070707071, 3.36276884658, 24.9832399071, 14.565909996, 57.4692063046, 45.5679137176, 36.8875136317, 34.8203572632, 18.3064850965]
f_score_untuned_ct = [86.1680327869, 33.3333333333, 44.4782372425, 38.2047205303, 42.9447392372, 55.0769230769, 36.0407876231, 49.9658038721, 44.9883449883, 10.0287384105, 25.4932734134, 21.5081416885, 61.7372696737, 45.5679137176, 38.0542272141, 31.8702553485, 19.6107298274]
f_score_tuned_rf = [86.1680327869, 33.3333333333, 40.2335209564, 52.190293742, 48.4236101451, 77.6470588235, 32.4236101451, 39.4476101424, 85.5234424632, 07.04038867633, 25.5292424632, 14.7121588089, 59.6580085941, 45.5679137176, 37.9979457176, 31.9612154881, 20.1432994881]
f_score_tuned_ct = [86.1680327869, 33.3333333333, 44.9686501106, 32.190293742, 43.6740802509, 57.6470588235, 34.8445439307, 76.171875, 86.5800865801, 16.1515380563, 27.1795153597, 18.0737985439, 61.7711181869, 55.5679137176, 38.398825739, 32.6086956522, 22.5436149896]

f_score_untuned_rf = nump.asarray(f_score_untuned_rf).astype(float)
f_score_tuned_rf = nump.asarray(f_score_tuned_rf).astype(float)
f_score_untuned_ct = nump.asarray(f_score_untuned_ct).astype(float)
f_score_tuned_ct = nump.asarray(f_score_tuned_ct).astype(float)

delta_f_score_rf = f_score_tuned_rf - f_score_untuned_rf
delta_f_score_ct = f_score_tuned_ct - f_score_untuned_ct

dataset = []

for i in range(1,18):
    dataset.append(i)

plt.title("Delta of F-Score between tuned and untuned Random Forest, CART")
plt.plot(dataset, delta_f_score_rf, color = 'g', label='Random Forest')
plt.plot(dataset, delta_f_score_ct, color = 'y', label='CART')
plt.legend()
plt.show()