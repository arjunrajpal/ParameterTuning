import matplotlib.pyplot as plt
import modified_SA_random_forest_precision_neighbour
import modified_SA_random_forest_precision_cauchy
import modified_SA_random_forest_precision_fast

import modified_SA_random_forest_fscore_neighbour
import modified_SA_random_forest_fscore_cauchy
import modified_SA_random_forest_fscore_fast

import modified_SA_CART_precision_neighbour
import modified_SA_CART_precision_cauchy
import modified_SA_CART_precision_fast

import modified_SA_CART_fscore_neighbour
import modified_SA_CART_fscore_cauchy
import modified_SA_CART_fscore_fast

import lr_vs_rf_vs_cart

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# Untuned Models

precision_cart_untuned, precision_rf_untuned, precision_lr_untuned, fscore_cart_untuned, fscore_rf_untuned, \
fscore_lr_untuned = lr_vs_rf_vs_cart.calculate_all()

print color.BOLD + color.RED + "Untuned Model" + color.END
print color.BOLD + color.CYAN + "Precision list for Untuned Cart" + color.END + str(precision_cart_untuned)
print color.BOLD + color.CYAN + "Fscore list for Untuned Cart " + color.END + str(fscore_cart_untuned)
print color.BOLD + color.CYAN + "Precision list for Untuned Random Forest" + color.END + str(precision_rf_untuned)
print color.BOLD + color.CYAN + "Fscore list for Untuned Random Forest " + color.END + str(fscore_rf_untuned)

# # Tuned Models

precision_cart_tuned_neighbour = modified_SA_CART_precision_neighbour.calculate()
precision_cart_tuned_cauchy = modified_SA_CART_precision_cauchy.calculate()
precision_cart_tuned_fast = modified_SA_CART_precision_fast.calculate()

fscore_cart_tuned_neighbour = modified_SA_CART_fscore_neighbour.calculate()
fscore_cart_tuned_cauchy = modified_SA_CART_fscore_cauchy.calculate()
fscore_cart_tuned_fast = modified_SA_CART_fscore_fast.calculate()

precision_rf_tuned_neighbour = modified_SA_random_forest_precision_neighbour.calculate()
precision_rf_tuned_cauchy = modified_SA_random_forest_precision_cauchy.calculate()
precision_rf_tuned_fast = modified_SA_random_forest_precision_fast.calculate()

fscore_rf_tuned_neighbour = modified_SA_random_forest_fscore_neighbour.calculate()
fscore_rf_tuned_cauchy = modified_SA_random_forest_fscore_cauchy.calculate()
fscore_rf_tuned_fast = modified_SA_random_forest_fscore_fast.calculate()

print color.BOLD + color.RED + "Tuned Model" + color.END
print color.BOLD + color.CYAN + "Precision list for Tuned Cart using random neighbour selection" + color.END + str(precision_cart_tuned_neighbour)
print color.BOLD + color.CYAN + "Precision list for Tuned Cart using cauchy schedule" + color.END + str(precision_cart_tuned_cauchy)
print color.BOLD + color.CYAN + "Precision list for Tuned Cart using fast schedule" + color.END + str(precision_cart_tuned_fast)

print color.BOLD + color.CYAN + "Fscore list for Tuned Cart using random neighbour selection" + color.END + str(fscore_cart_tuned_neighbour)
print color.BOLD + color.CYAN + "Fscore list for Tuned Cart using cauchy schedule" + color.END + str(fscore_cart_tuned_cauchy)
print color.BOLD + color.CYAN + "Fscore list for Tuned Cart using fast schedule" + color.END + str(fscore_cart_tuned_fast)

print color.BOLD + color.CYAN + "Precision list for Tuned Random Forest using random neighbour selection" + color.END + str(precision_rf_tuned_neighbour)
print color.BOLD + color.CYAN + "Precision list for Tuned Random Forest using cauchy schedule" + color.END + str(precision_rf_tuned_cauchy)
print color.BOLD + color.CYAN + "Precision list for Tuned Random Forest using fast schedule" + color.END + str(precision_rf_tuned_fast)

print color.BOLD + color.CYAN + "Fscore list for Tuned Random Forest using random neighbour selection" + color.END + str(fscore_rf_tuned_neighbour)
print color.BOLD + color.CYAN + "Fscore list for Tuned Random Forest using cauchy schedule" + color.END + str(fscore_rf_tuned_cauchy)
print color.BOLD + color.CYAN + "Fscore list for Tuned Random Forest using fast schedule" + color.END + str(fscore_rf_tuned_fast)

print "\n"

# Graph Plotting

data = []
for i in range(0,17):
    data.append(i+1)

# precision_cart_untuned = [82.12890, 25.0, 43.75047, 35.81006, 39.41437, 69.97899, 37.60467, 47.81746, 92.72727, 22.00978, 24.33392, 34.85078, 56.35903, 36.56030, 36.55555, 28.44961, 16.17081]

# precision_cart_tuned_cauchy = [82.12890625, 25.0, 31.3869937582, 23.9762187872, 28.8620416478, 74.4444444444, 21.9355872456, 31.640625, 77.9220779221, 19.8760393046, 16.7051888046, 9.64002341311,
# 51.1657653778, 36.5603028664, 19.0006574622, 10.165931527, 8.71397975588]

# precision_cart_tuned_fast = [82.12890625, 25.0, 31.3869937582, 23.9762187872, 41.4831100236, 49.0, 25.3030843288, 31.640625, 92.4242424242, 0.907029478458, 11.5533014633, 9.64002341311, 51.1657653778,
# 36.5603028664, 24.9206349206, 10.165931527, 19.035815416]

# precision_cart_tuned_neighbour = [82.12890625, 25.0, 31.3869937582, 23.9762187872, 28.8620416478, 49.0, 21.9355872456, 31.640625, 82.6446280992, 0.907029478458, 11.5533014633, 9.64002341311, 51.1657653778, 
# 36.5603028664, 19.0006574622, 10.165931527, 8.71397975588]

# fscore_cart_untuned = [86.16803, 33.33333, 44.47823, 38.20472, 42.94473, 55.07692, 36.04078, 49.96580, 44.98834, 10.02873, 25.49327, 21.50814, 61.73726, 45.56791, 38.05422, 31.87025, 19.61072]

# fscore_cart_tuned_cauchy = []

# fscore_cart_tuned_fast = []

# fscore_cart_tuned_neighbour = []

# precision_rf_untuned = [82.12890, 25.0, 38.30409, 35.20200, 40.94668, 77.35 ,28.28278, 36.82598, 79.54545, 7.99419, 22.11611, 9.55334, 48.57635, 36.56030, 34.65811, 29.34782, 14.79973]

# precision_rf_tuned_cauchy = [82.12890,25.0,38.30409,35.20200,40.94668,77.35,28.28278,36.82598,79.54545,7.99419,22.11611,9.55334,48.57635,36.56030,34.65811,29.34782,14.79973]

# precision_rf_tuned_fast = [82.12890625,25.0,31.3869937582,23.9762187872,28.8620416478,49.0,21.9355872456,31.640625,82.6446280992,0.907029478458,11.5533014633,9.64002341311,51.1657653778,36.5603028664,19.0006574622,10.165931527,8.71397975588]

# precision_rf_tuned_neighbour = [82.12890,25,41.38699,23.97621,58.32742,74.54545,36.21157,39.24557,92.99988,18.56332,24.56772,37.77988,51.16576,36.56030,32.56332,37.50374,15.48091]

# fscore_rf_untuned = [86.16803, 33.33333, 43.51349, 39.66459, 44.88794, 64.641241, 35.788262, 38.104838, 70.707070, 3.36276, 24.98323, 14.56590, 57.46920, 45.56791, 36.88751, 34.82035, 18.30648]

# fscore_rf_tuned_cauchy = [86.1680327869, 33.3333333333, 40.2335209564, 32.190293742, 37.550614739, 57.6470588235, 29.8777826277, 41.0472972973, 86.5800865801, 1.65631469979, 17.2450014489, 14.7121588089, 59.6580085941, 45.5679137176, 26.4652014652, 15.4164675904, 30.6858326707]

# fscore_rf_tuned_fast = [86.1680327869, 33.3333333333, 40.2335209564, 32.190293742, 39.7373058836, 57.6470588235, 30.4019542527, 45.6789023521, 86.5800865801, 1.65631469979, 17.2450014489, 14.7121588089, 59.6580085941, 45.5679137176, 26.4652014652, 15.4164675904, 19.638610593]

# fscore_rf_tuned_neighbour = [86.1680327869, 33.3333333333, 40.2335209564, 32.190293742, 37.550614739, 57.6470588235, 29.8777826277, 40.5, 86.5800865801, 1.65631469979, 17.2450014489, 14.7121588089, 59.6580085941, 45.5679137176, 26.4652014652, 15.4164675904, 13.4558627326]

plt.figure(1)
plt.title("Precision for Tuned vs Untuned Cart using Random Neighbour selection")
plt.plot(data, precision_cart_untuned, color='r', label='Untuned CART')
plt.plot(data, precision_cart_tuned_neighbour, color='g', label='Tuned CART')
plt.legend()

plt.figure(2)
plt.title("Precision for Tuned vs Untuned Cart using Cauchy Schedule")
plt.plot(data, precision_cart_untuned, color='r', label='Untuned CART')
plt.plot(data, precision_cart_tuned_cauchy, color='g', label='Tuned CART')
plt.legend()

plt.figure(3)
plt.title("Precision for Tuned vs Untuned Cart using Fast Schedule")
plt.plot(data, precision_cart_untuned, color='r', label='Untuned CART')
plt.plot(data, precision_cart_tuned_fast, color='g', label='Tuned CART')
plt.legend()

plt.figure(4)
plt.title("Precision for Tuned vs Untuned Random Forest using Random Neighbour selection")
plt.plot(data, precision_rf_untuned, color='r', label='Untuned Random Forest')
plt.plot(data, precision_rf_tuned_neighbour, color='g', label='Tuned Random Forest')
plt.legend()

plt.figure(5)
plt.title("Precision for Tuned vs Untuned Random Forest using Cauchy Schedule")
plt.plot(data, precision_rf_untuned, color='r', label='Untuned Random Forest')
plt.plot(data, precision_rf_tuned_cauchy, color='g', label='Tuned Random Forest')
plt.legend()

plt.figure(6)
plt.title("Precision for Tuned vs Untuned Random Forest using Fast Schedule")
plt.plot(data, precision_rf_untuned, color='r', label='Untuned Random Forest')
plt.plot(data, precision_rf_tuned_fast, color='g', label='Tuned Random Forest')
plt.legend()

plt.figure(7)
plt.title("Fscore for Tuned vs Untuned Cart using Random Neighbour selection")
plt.plot(data, fscore_cart_untuned, color='r', label='Untuned CART')
plt.plot(data, fscore_cart_tuned_neighbour, color='g', label='Tuned CART')
plt.legend()

plt.figure(8)
plt.title("Fscore for Tuned vs Untuned Cart using Cauchy Schedule")
plt.plot(data, fscore_cart_untuned, color='r', label='Untuned CART')
plt.plot(data, fscore_cart_tuned_cauchy, color='g', label='Untuned CART')
plt.legend()

plt.figure(9)
plt.title("Fscore for Tuned vs Untuned Cart using Fast Schedule")
plt.plot(data, fscore_cart_untuned, color='r', label='Untuned CART')
plt.plot(data, fscore_cart_tuned_fast, color='g', label='Tuned CART')
plt.legend()

plt.figure(10)
plt.title("Fscore for Tuned vs Untuned Random Forest using Random Neighbour selection")
plt.plot(data, fscore_rf_untuned, color='r', label='Untuned Random Forest')
plt.plot(data, fscore_rf_tuned_neighbour, color='g', label='Tuned Random Forest')
plt.legend()

plt.figure(11)
plt.title("Fscore for Tuned vs Untuned Random Forest using Cauchy Schedule")
plt.plot(data, fscore_rf_untuned, color='r', label='Untuned Random Forest')
plt.plot(data, fscore_rf_tuned_cauchy, color='g', label='Untuned Random Forest')
plt.legend()

plt.figure(12)
plt.title("Fscore for Tuned vs Untuned Random Forest using Fast Schedule")
plt.plot(data, fscore_rf_untuned, color='r', label='Untuned Random Forest')
plt.plot(data, fscore_rf_tuned_fast, color='g', label='Tuned Random Forest')
plt.legend()

plt.show()