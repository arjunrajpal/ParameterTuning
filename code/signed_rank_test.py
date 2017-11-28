import csv
import numpy as np

def signed_rank(before_tuning, after_tuning):
	# with open('results.csv','r') as f:
	# reader = csv.reader(f)
	# data = list(reader)

	# data = np.asarray(data).astype(np.integer)

	# beforeTuning  = data[:,0]
	# afterTuning = data[:,1]

	# difference = beforeTuning-afterTuning

	# # difference = sorted(difference)

	# print difference

	l = []
	# before_tuning = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
	# after_tuning = [10,20,30,40,34,89,12,34,25,87,46,67,52,1,23,32,39]

	for i in range(len(before_tuning)):
		d={}
		d['before_tuning'] = before_tuning[i]
		d['after_tuning'] = after_tuning[i]
		d['difference'] = after_tuning[i]-before_tuning[i]
		d['absolute_value'] = abs(d['difference'])

		if d['difference']>=0:
			d['sign'] = '+'
		else:
			d['sign'] = '-'

		l.append(d)

	# print l

	l = sorted(l, key=lambda k: k['absolute_value']) 

	for i in range(len(l)):
		l[i]['rank'] = i+1

	# print l[0:3]
	for i in range(len(l)):
		print l[i]['difference']

	CRITICAL_VALUE = 41

	sum_pos = 0
	sum_neg = 0

	for i in range(len(l)):
		if l[i]['sign']=='+':
			sum_pos += l[i]['rank']
		else:
			sum_neg += l[i]['rank']

	print "sum_pos ",str(sum_pos),"sum_neg ",str(sum_neg)

	min_sum = min(sum_pos,sum_neg)

	if min_sum < CRITICAL_VALUE:
		print "Null hypothesis rejected, thus they are different"
		# return "Null hypothesis rejected, thus they are different"
	else:
		print "Null hypothesis is accepted, thus they are same"
		# return "Null hypothesis is accepted, thus they are same"

signed_rank(before_tuning = [82.12890,25.0,38.30409,35.20200,40.94668,77.35,28.28278,36.82598,79.54545,7.99419,22.11611,9.55334,48.57635,36.56030,34.65811,29.34782,14.79973],	after_tuning = [82.12890625,25.0,31.3869937582,23.9762187872,28.8620416478,49.0,21.9355872456,31.640625,82.6446280992,0.907029478458,11.5533014633,9.64002341311,51.1657653778,36.5603028664,19.0006574622,10.165931527,8.71397975588])
signed_rank(before_tuning = [82.12890,25.0,38.30409,35.20200,40.94668,77.35,28.28278,36.82598,79.54545,7.99419,22.11611,9.55334,48.57635,36.56030,34.65811,29.34782,14.79973],	after_tuning = [82.12890,25,41.38699,23.97621,58.32742,74.54545,36.21157,39.24557,92.99988,18.56332,24.56772,37.77988,51.16576,36.56030,32.56332,37.50374,15.48091])

# difference_array = [] 
# for i in range(len(difference)):
# 	if difference[i] >= 0:
# 		sign = '+'
# 	else:
# 		sign = '-'
	
# 	d = {'absolute_value':abs(difference[i]), 'sign':sign, 'rank':0}

# 	difference_array.append(d)

# difference_array = sorted(difference_array)

# sum = difference_array[0]['absolute_value']
# start = 0
# end = 0

# start_rank = 1
# difference_array[0] = start_rank

# for i in range(len(difference_array)-1):
# 	if difference_array[i]['absolute_value'] == difference_array[i+1]['absolute_value']:
# 		difference_array[i+1]['rank'] = start_rank
# 	else:
# 		start_rank = start_rank+1
# 		difference_array[i+1]['rank'] = start_rank

	
# for i in range(len(difference_array)-1):
# 	if difference_array[i]['absolute_value'] == difference_array[i+1]['absolute_value']:
# 		sum += difference_array[i]['absolute_value']
# 		end = i+1
# 	else:
# 		for j in range(end-start+1):
# 			difference_array[j]['rank'] = sum/(end-start+1)
# 		start = i+1
# 		end = i+1
# 		sum = difference_array[start]['absolute_value']

# for j in range(end-start+1):
# 	difference_array[j]['rank'] = sum/(end-start+1)

# print difference_array


