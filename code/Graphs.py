import matplotlib.pyplot as plt
import lr_vs_rf_vs_cart
import DE_randomforest_precision
import DE_randomforest_fscore
import DE_cart_precision
import DE_cart_fscore


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

# Tuned Models

#precision_rf_tuned = DE_randomforest_precision.calculate()
#fscore_rf_tuned = DE_randomforest_fscore.calculate()

precision_cart_tuned = DE_cart_precision.calculate()
fscore_cart_tuned = DE_cart_fscore.calculate()

print color.BOLD + color.RED + "Tuned Model" + color.END
# print color.BOLD + color.CYAN + "Precision list for Tuned Random Forest" + color.END + str(precision_rf_tuned)
# print color.BOLD + color.CYAN + "Fscore list for Tuned Random Forest" + color.END + str(fscore_rf_tuned)
print color.BOLD + color.CYAN + "Precision list for Tuned Cart" + color.END + str(precision_cart_tuned)
print color.BOLD + color.CYAN + "Fscore list for Tuned Cart " + color.END + str(fscore_cart_tuned)

print "\n"

# Graph Plotting

data = []
for i in range(0,17):
    data.append(i+1)

delta_cart_precision = []
for i in range(0,17):
    delta_cart_precision.append(abs(precision_cart_tuned[i] - precision_cart_untuned[i]))

# delta_rf_precision = []
# for i in range(0,17):
#     delta_rf_precision.append(abs(precision_rf_tuned[i] - precision_rf_untuned[i]))

delta_cart_fscore = []
for i in range(0, 17):
   delta_cart_fscore.append(abs(fscore_cart_tuned[i] - fscore_cart_untuned[i]))

# delta_rf_fscore = []
# for i in range(0, 17):
#    delta_rf_fscore.append(abs(fscore_rf_tuned[i] - fscore_rf_untuned[i]))

plt.figure(1)
plt.title("Precision for Untuned - Cart, Random Forest and Logistic Regression")
plt.plot(data, precision_cart_untuned, color='r', label='Untuned CART')
plt.plot(data, precision_rf_untuned, color='g', label='Untuned Random Forest')
plt.plot(data, precision_lr_untuned, color='b', label='Untuned Logistic Regression')
plt.legend()

plt.figure(2)
plt.title("Fscore for Untuned - Cart, Random Forest and Logistic Regression")
plt.plot(data, fscore_cart_untuned, color='r', label='Untuned CART')
plt.plot(data, fscore_rf_untuned, color='g', label='Untuned Random Forest')
plt.plot(data, fscore_lr_untuned, color='b', label='Untuned Logistic Regression')
plt.legend()

plt.figure(3)
plt.title("Precision for Tuned vs Untuned Cart")
plt.plot(data, precision_cart_untuned, color='r', label='Untuned CART')
plt.plot(data, precision_cart_tuned, color='g', label='Tuned CART')
plt.legend()

plt.figure(4)
plt.title("Fscore for Tuned vs Untuned Cart")
plt.plot(data, fscore_cart_untuned, color='r', label='Untuned CART')
plt.plot(data, fscore_cart_tuned, color='g', label='Tuned CART')
plt.legend()

plt.figure(5)
plt.title("Delta in precision for Cart")
plt.plot(data, delta_cart_precision, color='r')
plt.legend()

plt.figure(6)
plt.title("Delta in Fscore for Cart")
plt.plot(data, delta_cart_fscore, color='r')
plt.legend()

# plt.figure(7)
# plt.title("Precision for Tuned vs Untuned Random Forest")
# plt.plot(data, precision_rf_untuned, color='r', label='Untuned Random Forest')
# plt.plot(data, precision_rf_tuned, color='g', label='Tuned Random Forest')
# plt.legend()
#
# plt.figure(8)
# plt.title("Fscore for Tuned vs Untuned Random Forest")
# plt.plot(data, fscore_rf_untuned, color='r', label='Untuned Random Forest')
# plt.plot(data, fscore_rf_tuned, color='g', label='Tuned Random Forest')
# plt.legend()
#
# plt.figure(9)
# plt.title("Delta in precision for Random Forest")
# plt.plot(data, delta_rf_precision, color='r')
# plt.legend()
#
# plt.figure(10)
# plt.title("Delta in Fscore for Random Forest")
# plt.plot(data, delta_rf_fscore, color='r')
# plt.legend()

plt.show()