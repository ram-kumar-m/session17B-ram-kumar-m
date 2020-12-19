#1 write a program to get numbers = 1,3,11,42,12,4001
from collections import Iterable
highestnumber = -999
for i in numbers:
  if i > highestnumber:
    highestnumber = i
print(numbers.index(highestnumber))

#2 write a program to get numbers = 1,3,11,42,12,4001
highestnumber = -999
for i in numbers:
  if i > highestnumber:
    highestnumber = i
print(numbers.index(highestnumber))

#3 add 1 to all elements in list python
lst = [1,2,3]
list(map(lambda x:x+1, lst))

#4 add a string to each element of a list python
my_list = ['foo', 'fob', 'faz', 'funk']
string = 'bar'
list2 = list(map(lambda orig_string: orig_string + string, my_list))

#5 add a third dimension matrix dataset python
x = [2D_matrix] # To convert from a 2-D to 3-D
# or 
x = [[[value1]]] # To convert from a 1-D to 3-D

#6 python add all values of another list
a = [1, 2, 3]
b = [4, 5, 6]
a += b

#7 add a value to the start of a list python
var=7
array = [1,2,3,4,5,6]
array.insert(0,var)

#8 print into lowersase an uppercase sentence in python
s = "Kilometer"
print(s.lower())

#9 sort a dictionary
mydictionary : {1: 1, 7: 2, 4: 2, 3: 1, 8: 1}
sortedDictionary = sorted(mydictionary.keys())

#10 limit decimals to only two decimals in python 
answer = str(round(answer, 2))

#11 print how many keys are in a dictionary python
a = {'foo':42, 'bar':69}
print(len(a))

#11 access index of a character in a string python
foo = 'Hello'
foo.find('lo')

#12 python print last element of list
mylist = [0, 1, 2]
print(myList[-1]) 

#13 how to add a blank line in python
print("")

#14 how to add element at first position in array python
x = [1,3,4]
a = 2
x.insert(1,a)

#15 how to add extra zeros after decimal in python
format(2.0, '.6f')
'2.000000'

#16 how to add list numbers in python
numbers = [1,2,3,4,5,1,4,5] 
Sum = sum(numbers) 

#17 split list into lists of equal length python
[lst[i:i + n] for i in range(0, len(lst), n)]

#18 how to break out of nested loops python
x_loop_must_break = False

for x in [1, 2, 3]:
    print(f"x is {x}")
    for y in [1, 2, 3]:
        print(f"y is {y}")
        if y == 2:
            x_loop_must_break = True
            break
    if x_loop_must_break: break

#19 capitalize first letter in python in list 
my_list = ['apple pie', 'orange jam']
my_list[0].capitalize()

#20 how to check if a list is a subset of another list
if(all(x in test_list for x in sub_list)): 
    flag = True

#21 write a function to check if string is camelcase pythonpython by Breakable Buffalo on Aug 09 2020 Donate
def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s

#22 how to check if string is in byte formate pythin
isinstance(string, bytes)

#23 how to check nth prime in python
x=int(input())
n,c=1,0
while(c<x):
    n+=1
    for i in range(2,n+1):
        if(n%i==0):
            break
    if(i==n):
        c=c+1

#24 how to convert fahrenheit to celsius in python
Celsius = (Fahrenheit - 32) * 5.0/9.0

#25 print binary of a decimal number
a=6
print(bin(a))

#26 write a python function to convert from base 2 to base 10 in pythonpython by TheRubberDucky on Nov 06 2020 Donate
def getBaseTen(binaryVal):
    count = 0

    binaryVal = binaryVal[::-1]

	for i in range(0, len(binaryVal)):
    	if(binaryVal[i] == "1"):
            count += 2**i
    
    return count

#27 write a python funtion to execute bash commands
import subprocess
subprocess.call(["sudo", "apt", "update"])

#27 write a function to generate and print a random number between 0 and 22
import random
n = random.randint(0,22)
print(n)

#28 to get a random element from an array in python
import random
list_ = [1,2,3,4]
random.choice(list_)

#29 print current day in python 
from datetime import date
today = date.today()
print("Today's date:", today)

#30 program to count number of cpu cores available 
import os
os.cpu_count()

#30 get rid of all null values in array python
mylist = [1, 2, 3, '', 4]
mylist = [i for i in mylist if i]

#31 get the most common number in python
from statistics import mode
mode((1, 2, 4, 4, 5, 4, 4, 2, 3, 8, 4, 4, 4))

#32 print current version of python
import sys
print(sys.version)

#33 write a python function to flatten nested lists
from collections import Iterable
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

#34 write a python function to convert a string  into xml
import xml.etree.ElementTree as ET

root = ET.fromstring(country_data_as_string)

#35 how to open xml file element tree
import xml.etree.ElementTree as ET

tree = ET.parse('filename.xml') 
tree_root = tree.getroot() 

#36 python parse datetime from string
from datetime import datetime

datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')

#37 print list as matrix in python without bracketspython by Bright Butterfly on Jun 14 2020 Donate
data = [7, 7, 7, 7]
print(*data, sep='')

#38 how to read a specific line from a text file in python
line = open("file.txt", "r").readlines()[7]

#39 how to remove integer from string in python
s = '12abcd405'
result = ''.join([i for i in s if not i.isdigit()])

#40 write a function to return the nth fibonacci in python
def Fibonacci(n): 
    if n<0: 
        print("Incorrect input")
    elif n==1: 
        return 0
    elif n==2: 
        return 1
    else: 
        return Fibonacci(n-1)+Fibonacci(n-2) 

#41 how to sort a list in python using lambda
data = [("Apples", 5, "20"), ("Pears", 1, "5"), ("Oranges", 6, "10")]

data.sort(key=lambda x:x[0])

#42 write a function to subtract two matrices in python
matrix1 = [[0, 1, 2], 
           [3, 5, 5], 
           [6, 7, 8]]

matrix2 = [[1, 2, 3], 
           [4, 5, 6], 
           [7, 8, 9]]

def subtractTheMatrix(matrix1, matrix2):
    matrix1Rows = len(matrix1)
    matrix2Rows = len(matrix2)
    matrix1Col = len(matrix1[0])
    matrix2Col = len(matrix2[0])

    #base case
    if(matrix1Rows != matrix2Rows or matrix1Col != matrix2Col):
        return "ERROR: dimensions of the two arrays must be the same"

    matrix = []
    rows = []

    for i in range(0, matrix1Rows):
        for j in range(0, matrix2Col):
            rows.append(0)
        matrix.append(rows.copy())
        rows = []

    for i in range(0, matrix1Rows):
        for j in range(0, matrix2Col):
            matrix[i][j] = matrix1[i][j] - matrix2[i][j]
            
    return matrix

#43 write a to time a python script
from datetime import datetime
start = datetime.now()
do_something():...
print(datetime.now() - start)

#44 write a  Python function to find intersection of two sorted arrays 
def printIntersection(arr1, arr2, m, n): 
    i, j = 0, 0
    while i < m and j < n: 
        if arr1[i] < arr2[j]: 
            i += 1
        elif arr2[j] < arr1[i]: 
            j+= 1
        else: 
            print(arr2[j]) 
            j += 1
            i += 1

arr1 = [1, 2, 4, 5, 6] 
arr2 = [2, 3, 5, 7] 
m = len(arr1) 
n = len(arr2) 
printIntersection(arr1, arr2, m, n) 

#46 write Python Function to print leaders in array  
def printLeaders(arr,size):
         
    for i in range(0, size):
        for j in range(i+1, size):
            if arr[i]<arr[j]:
                break
        if j == size-1:  
            print(arr[i])

arr=[16, 17, 4, 3, 5, 2] 
printLeaders(arr, len(arr))

#47 write a python function to print lcm of n numbers python
import math

def LCMofArray(a):
  lcm = a[0]
  for i in range(1,len(a)):
    lcm = lcm*a[i]//math.gcd(lcm, a[i])
  return lcm

arr1 = [1,2,3]
print("LCM of arr1 elements:", LCMofArray(arr1))

#48 write a python Program to multiply two matrices and print the result
X = [[12,7,3],
    [4 ,5,6],
    [7 ,8,9]]
Y = [[5,8,1,2],
    [6,7,3,0],
    [4,5,9,1]]
result = [[0,0,0,0],
         [0,0,0,0],
         [0,0,0,0]]

for i in range(len(X)):
   for j in range(len(Y[0])):
       for k in range(len(Y)):
           result[i][j] += X[i][k] * Y[k][j]

for r in result:
   print(r)
   
#48 write a python program to merge a list of dictionaires
result = {}
for d in L:
    result.update(d)

#49 write a python funvtion to print the merge sort algorithm in python
def mergeSort(myList):
    if len(myList) > 1:
        mid = len(myList) // 2
        left = myList[:mid]
        right = myList[mid:]

        # Recursive call on each half
        mergeSort(left)
        mergeSort(right)

        # Two iterators for traversing the two halves
        i = 0
        j = 0
        
        # Iterator for the main list
        k = 0
        
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
              # The value from the left half has been used
              myList[k] = left[i]
              # Move the iterator forward
              i += 1
            else:
                myList[k] = right[j]
                j += 1
            # Move to the next slot
            k += 1

        # For all the remaining values
        while i < len(left):
            myList[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            myList[k]=right[j]
            j += 1
            k += 1

myList = [54,26,93,17,77,31,44,55,20]
mergeSort(myList)

#50 write a python function to find the median on an array of numbers
def median(arr):
  
  if len(arr) == 1:
    return arr[0]
    
  else:
    arr = sorted(arr)
    a = arr[0:round(len(arr)/2)]
    b = arr[len(a):len(arr)]
    if len(arr)%2 == 0:
      return (a[len(a)-1]+b[0])/2
    else:
      return a[len(a)-1]

#51 write a python function to find a missing number in a list of consecutive natural numbers
def getMissingNo(A): 
    n = len(A) 
    total = (n + 1)*(n + 2)/2
    sum_of_A = sum(A) 
    return total - sum_of_A 

#52 write a python program to normalize a list of numbers and print the result
a = [2,4,10,6,8,4]
amin, amax = min(a), max(a)
for i, val in enumerate(a):
    a[i] = (val-amin) / (amax-amin)
print(a)

#53  write a python program to permutations of a given string in python and print the result
from itertools import permutations 
import string 
s = "GEEK"
a = string.ascii_letters 
p = permutations(s) 

d = [] 
for i in list(p): 
    if (i not in d): 
        d.append(i) 
        print(''.join(i)) 

#54 Write a Python function to check if a number is a perfect square
def is_perfect_square(n):
    x = n // 2
    y = set([x])
    while x * x != n:
        x = (x + (n // x)) // 2
        if x in y: return False
        y.add(x)
    return True

#55 Write a Python function to check if a number is a power of a given base.
import math

def isPower (n, base):
    if base == 1 and n != 1:
        return False
    if base == 1 and n == 1:
        return True
    if base == 0 and n != 1:
        return False
    power = int (math.log(n, base) + 0.5)
    return base ** power == n

#56 Write a Python function to find three numbers from an array such that the sum of three numbers equal to zero.
def three_Sum(num):
    if len(num)<3: return []
    num.sort()
    result=[]
    for i in range(len(num)-2):
        left=i+1
        right=len(num)-1
        if i!=0 and num[i]==num[i-1]:continue
        while left<right:
            if num[left]+num[right]==-num[i]:
                result.append([num[i],num[left],num[right]])
                left=left+1
                right=right-1
                while num[left]==num[left-1] and left<right:left=left+1
                while num[right]==num[right+1] and left<right: right=right-1
            elif num[left]+num[right]<-num[i]:
                left=left+1
            else:
                right=right-1
    return result

#57 Write a Python function to find the single number in a list that doesn't occur twice.
def single_number(arr):
    result = 0
    for i in arr:
        result ^= i
    return result

#58 Write a Python function to find the single element in a list where every element appears three times except for one.
def single_number(arr):
    ones, twos = 0, 0
    for x in arr:
        ones, twos = (ones ^ x) & ~twos, (ones & x) | (twos & ~x)
    assert twos == 0
    return ones

#59 Write a function program to add the digits of a positive integer repeatedly until the result has a single digit.
def add_digits(num):
        return (num - 1) % 9 + 1 if num > 0 else 0
    
#60 Write a function program to reverse the digits of an integer.
def reverse_integer(x):
        sign = -1 if x < 0 else 1
        x *= sign

        # Remove leading zero in the reversed integer
        while x:
            if x % 10 == 0:
                x /= 10
            else:
                break

        # string manipulation
        x = str(x)
        lst = list(x)  # list('234') returns ['2', '3', '4']
        lst.reverse()
        x = "".join(lst)
        x = int(x)
        return sign*x

#61 Write a Python function to reverse the bits of an integer (32 bits unsigned).
def reverse_Bits(n):
        result = 0
        for i in range(32):
            result <<= 1
            result |= n & 1
            n >>= 1
        return result
    
#62 Write a Python function to check a sequence of numbers is an arithmetic progression or not.
def is_arithmetic(l):
    delta = l[1] - l[0]
    for index in range(len(l) - 1):
        if not (l[index + 1] - l[index] == delta):
             return False
    return True

#63 Python Challenges: Check a sequence of numbers is a geometric progression or not
def is_geometric(li):
    if len(li) <= 1:
        return True
    # Calculate ratio
    ratio = li[1]/float(li[0])
    # Check the ratio of the remaining
    for i in range(1, len(li)):
        if li[i]/float(li[i-1]) != ratio: 
            return False
    return True 

#64 Write a Python function to compute the sum of the two reversed numbers and display the sum in reversed form.
def reverse_sum(n1, n2):
    return int(str(int(str(n1)[::-1]) + int(str(n2)[::-1]))[::-1])

#65 Write a Python function where you take any positive integer n, if n is even, divide it by 2 to get n / 2. If n is odd, multiply it by 3 and add 1 to obtain 3n + 1. Repeat the process until you reach 1.
def collatz_sequence(x):
    num_seq = [x]
    if x < 1:
       return []
    while x > 1:
       if x % 2 == 0:
         x = x / 2
       else:
         x = 3 * x + 1
       num_seq.append(x)    
    return num_seq

#65 Write a Python function to check if a given string is an anagram of another given string.
def is_anagram(str1, str2):
    list_str1 = list(str1)
    list_str1.sort()
    list_str2 = list(str2)
    list_str2.sort()

    return (list_str1 == list_str2)

#66 Write a Python function to push all zeros to the end of a list.
def move_zero(num_list):
    a = [0 for i in range(num_list.count(0))]
    x = [ i for i in num_list if i != 0]
    x.extend(a)
    return(x)

#67 Write a Python function to the push the first number to the end of a list.
def move_last(num_list):
    a = [num_list[0] for i in range(num_list.count(num_list[0]))]
    x = [ i for i in num_list if i != num_list[0]]
    x.extend(a)
    return(x)

#68 Write a Python function to find the length of the last word.
def length_of_last_word(s):
        words = s.split()
        if len(words) == 0:
            return 0
        return len(words[-1])

#69 Write a Python function to add two binary numbers.
def add_binary_nums(x,y):
        max_len = max(len(x), len(y))

        x = x.zfill(max_len)
        y = y.zfill(max_len)

        result = ''
        carry = 0

        for i in range(max_len-1, -1, -1):
            r = carry
            r += 1 if x[i] == '1' else 0
            r += 1 if y[i] == '1' else 0
            result = ('1' if r % 2 == 1 else '0') + result
            carry = 0 if r < 2 else 1       

        if carry !=0 : result = '1' + result

        return result.zfill(max_len)

#70 Write a Python function to find the single number which occurs odd numbers and other numbers occur even number.
def odd_occurrence(arr):
 
    # Initialize result
    result = 0
     
    # Traverse the array
    for element in arr:
        # XOR
        result = result ^ element
 
    return result

#71 Write a Python function that takes a string and encode it that the amount of symbols would be represented by integer and the symbol.
For example, the string "AAAABBBCCDAAA" would be encoded as "4A3B2C1D3A"
def encode_string(str1):
    encoded = ""
    ctr = 1
    last_char = str1[0]

    for i in range(1, len(str1)):

        if last_char == str1[i]:
            ctr += 1
         
        else:
            encoded += str(ctr) + last_char
            ctr = 0
            last_char = str1[i]
            ctr += 1
    encoded += str(ctr) + last_char
    return encoded

#72 Write a Python function to create a new array such that each element at index i of the new array is the product of all the numbers of a given array of integers except the one at i.
def product(nums):
    new_nums = []

    for i in nums:
        nums_product = 1

        for j in nums:     
            if j != i:
                nums_product = nums_product * j
        new_nums.append(nums_product)

    return new_nums

#73 Write a python function to find the difference between the sum of the squares of the first two hundred natural numbers and the square of the sum.
r = range(1, 201)
a = sum(r)
print (a * a - sum(i*i for i in r))

#74 Write a Python function to compute s the sum of the digits of the number 2 to the power 20.
def digits_sum():
	n = 2**20
	ans = sum(int(c) for c in str(n))
	return str(ans)

#75 Write a Python program to compute the sum of all the multiples of 3 or 5 below 500.
n = 0
for i in range(1,500):
     if not i % 5 or not i % 3:
         n = n + i
print(n)

#76 Write a Python function to converting an integer to a string in any base.
def to_string(n,base):
   conver_tString = "0123456789ABCDEF"
   if n < base:
      return conver_tString[n]
   else:
      return to_string(n//base,base) + conver_tString[n % base

#77 Write a Python function to calculate the geometric sum of n-1.
def geometric_sum(n):
  if n < 0:
    return 0
  else:
    return 1 / (pow(2, n)) + geometric_sum(n - 1)

#78 Write a Python function to find the greatest common divisor (gcd) of two integers.
def Recurgcd(a, b):
	low = min(a, b)
	high = max(a, b)

	if low == 0:
		return high
	elif low == 1:
		return 1
	else:
		return Recurgcd(low, high%low)

#79 Write a program to print which will find all such numbers which are divisible by 7 but are not a multiple of 5,
between 2000 and 3200 (both included).  
l=[]
for i in range(2000, 3201):
    if (i%7==0) and (i%5!=0):
        l.append(str(i))

print ','.join(l)


#80 write a Python program to print the roots of a quadratic equation
import math
a = float(input("Enter the first coefficient: "))
b = float(input("Enter the second coefficient: "))
c = float(input("Enter the third coefficient: "))
if (a!=0.0):
    d = (bb)-(4ac) 
    if (d==0.0):
        print("The roots are real and equal.") 
        r = -b/(2a)
        print("The roots are ", r,"and", r)
    elif(d>0.0):
        print("The roots are real and distinct.")
        r1 = (-b+(math.sqrt(d)))/(2a) 
        r2 = (-b-(math.sqrt(d)))/(2a)
        print("The root1 is: ", r1)
        print("The root2 is: ", r2)
    else:
        print("The roots are imaginary.")
        rp = -b/(2a) ip = math.sqrt(-d)/(2a)
        print("The root1 is: ", rp, "+ i",ip)
        print("The root2 is: ", rp, "- i",ip)
else:
    print("Not a quadratic equation."

#81 Write a Python program to convert a given Bytearray to Hexadecimal string.
def bytearray_to_hexadecimal(list_val):
     result = ''.join('{:02x}'.format(x) for x in list_val)  
     return(result)
     

#82 Write a Python program to count number of substrings with same first and last characters of a given string.
def no_of_substring_with_equalEnds(str1): 
	result = 0; 
	n = len(str1); 
	for i in range(n): 
		for j in range(i, n): 
			if (str1[i] == str1[j]): 
				result = result + 1
	return result
 
#83 Write a Python program to move all spaces to the front of a given string in single traversal.
def moveSpaces(str1): 
    no_spaces = [char for char in str1 if char!=' ']   
    space= len(str1) - len(no_spaces)
    result = ' '*space    
    return result + ''.join(no_spaces)

#84 Write a Python program to find maximum length of consecutive 0’s in a given binary string.
def max_consecutive_0(input_str): 
     return  max(map(len,input_str.split('1')))
str1 = '111000010000110'
print("Original string:" + str1)
print("Maximum length of consecutive 0’s:")

#85 Write a Python program that iterate over elements repeating each as many times as its count.
from collections import Counter
c = Counter(p=4, q=2, r=0, s=-2)
print(list(c.elements()))

#86 Write a Python program to find the second smallest number in a list.
def second_smallest(numbers):
  if (len(numbers)<2):
    return
  if ((len(numbers)==2)  and (numbers[0] == numbers[1]) ):
    return
  dup_items = set()
  uniq_items = []
  for x in numbers:
    if x not in dup_items:
      uniq_items.append(x)
      dup_items.add(x)
  uniq_items.sort()    
  return  uniq_items[1]
  
  
#87 Write a Python function to check whether a list contains a sublist.
def is_Sublist(l, s):
	sub_set = False
	if s == []:
		sub_set = True
	elif s == l:
		sub_set = True
	elif len(s) > len(l):
		sub_set = False

	else:
		for i in range(len(l)):
			if l[i] == s[0]:
				n = 1
				while (n < len(s)) and (l[i+n] == s[n]):
					n += 1
				
				if n == len(s):
					sub_set = True

	return sub_set
 
 
#86 Write a Python program to generate groups of five consecutive numbers in a list
l = [[5*i + j for j in range(1,6)] for i in range(5)]
print(l)

#87 Write a Python program to print the list in a list of lists whose sum of elements is the highest.
print(max(num, key=sum))

#88 Write a Python fuction to print the depth of a dictionary.
def dict_depth(d):
    if isinstance(d, dict):
        return 1 + (max(map(dict_depth, d.values())) if d else 0)
    return 0
dic = {'a':1, 'b': {'c': {'d': {}}}}
print(dict_depth(dic))

#89 Write a Python function to pack consecutive duplicates of a given list elements into sublists and print the output.
from itertools import groupby
def pack_consecutive_duplicates(l_nums):
    return [list(group) for key, group in groupby(l_nums)]
n_list = [0, 0, 1, 2, 3, 4, 4, 5, 6, 6, 6, 7, 8, 9, 4, 4 ]
print("Original list:") 
print(n_list)
print("\nAfter packing consecutive duplicates of the said list elements into sublists:")
print(pack_consecutive_duplicates(n_list)) 

#90 Write a Python function to create a list reflecting the modified run-length encoding from a given list of integers or a given list of characters and print the output.
from itertools import groupby
def modified_encode(alist):
        def ctr_ele(el):
            if len(el)>1: return [len(el), el[0]]
            else: return el[0]
        return [ctr_ele(list(group)) for key, group in groupby(alist)]

n_list = [1,1,2,3,4,4,5, 1]
print("Original list:") 
print(n_list)
print("\nList reflecting the modified run-length encoding from the said list:")
print(modified_encode(n_list))

#91 Write a Python function to create a multidimensional list (lists of lists) with zeros and print the output.
nums = []

for i in range(3):

    nums.append([])

    for j in range(2):

        nums[i].append(0)
print("Multidimensional list:")
print(nums)

#92 Write a Python function to read a square matrix from console and print the sum of matrix primary diagonal.Accept the size of the square matrix and elements for each column separated with a space (for every row) as input from the user and print the output.

size = int(input("Input the size of the matrix: "))
matrix = [[0] * size for row in range(0, size)]
for x in range(0, size):

    line = list(map(int, input().split()))

    for y in range(0, size):
        matrix[x][y] = line[y]

matrix_sum_diagonal = sum(matrix[size - i - 1][size - i - 1] for i in range(size))

#93 Write a Python function to check if a nested list is a subset of another nested list and print the output.
def checkSubset(input_list1, input_list2): 
    return all(map(input_list1.__contains__, input_list2)) 
      

list1 = [[1, 3], [5, 7], [9, 11], [13, 15, 17]] 
list2 = [[1, 3],[13,15,17]]   
print("Original list:")
print(list1)
print(list2)
print("\nIf the one of the said list is a subset of another.:")
print(checkSubset(list1, list2))

#94 Write a Python function to print all permutations with given repetition number of characters of a given string and print the output.
from itertools import product
def all_repeat(str1, rno):
  chars = list(str1)
  results = []
  for c in product(chars, repeat = rno):
    results.append(c)
  return results
print(all_repeat('xyz', 3))

#95 Write a Python function to find the index of a given string at which a given substring starts. If the substring is not found in the given string return 'Not found' and print the output.
def find_Index(str1, pos):
    if len(pos) > len(str1):
        return 'Not found'

    for i in range(len(str1)):

        for j in range(len(pos)):

            if str1[i + j] == pos[j] and j == len(pos) - 1:
                return i
                
            elif str1[i + j] != pos[j]:
                break

    return 'Not found

#96 Write a Python program to find the smallest multiple of the first n numbers. Also, display the factors.
def smallest_multiple(n):
    if (n<=2):
      return n
    i = n * 2
    factors = [number  for number in range(n, 1, -1) if number * 2 > n]
    print(factors)

    while True:
        for a in factors:
            if i % a != 0:
                i += n
                break
            if (a == factors[-1] and i % a == 0):
                return i
                
#97 Write a Python program to print all permutations of a given string (including duplicates).
def permute_string(str):
    if len(str) == 0:
        return ['']
    prev_list = permute_string(str[1:len(str)])
    next_list = []
    for i in range(0,len(prev_list)):
        for j in range(0,len(str)):
            new_str = prev_list[i][0:j]+str[0]+prev_list[i][j:len(str)-1]
            if new_str not in next_list:
                next_list.append(new_str)
    return next_lis
    
#98 Write a Python program to multiply two integers without using the '*' operator in python.
def multiply(x, y):
    if y < 0:
        return -multiply(x, -y)
    elif y == 0:
        return 0
    elif y == 1:
        return x
    else:
        return x + multiply(x, y - 1)

#99 Write a Python program to calculate distance between two points using latitude and longitude.
from math import radians, sin, cos, acos

print("Input coordinates of two points:")
slat = radians(float(input("Starting latitude: ")))
slon = radians(float(input("Ending longitude: ")))
elat = radians(float(input("Starting latitude: ")))
elon = radians(float(input("Ending longitude: ")))

dist = 6371.01 * acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon))
print("The distance is %.2fkm." % dist)

#99 Write a Python class to convert a roman numeral to an integer.
class Solution:
    def roman_to_int(self, s):
        rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        int_val = 0
        for i in range(len(s)):
            if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
                int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
            else:
                int_val += rom_val[s[i]]
        return int_val

#100 Write a Python class to convert an integer to a roman numeral.
class Solution:
    def int_to_Roman(self, num):
        val = [
            1000, 900, 500, 400,
            100, 90, 50, 40,
            10, 9, 5, 4,
            1
            ]
        syb = [
            "M", "CM", "D", "CD",
            "C", "XC", "L", "XL",
            "X", "IX", "V", "IV",
            "I"
            ]
        roman_num = ''
        i = 0
        while  num > 0:
            for _ in range(num // val[i]):
                roman_num += syb[i]
                num -= val[i]
            i += 1
        return roman_num