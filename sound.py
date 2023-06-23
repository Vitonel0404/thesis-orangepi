import os
import random
from pygame import mixer


dir = "/home/victor/VictorIA/app-thesis-master/sounds/"
archivos = os.listdir(dir)

def playSound():

	count = 0
	arrayNumber = []
	arrayName = []
	
	for x in archivos:
		if x != 'test.mp3':
			arrayName.append(x)
			arrayNumber.append(int(x.split('-')[0]))

	arrayNumber.sort()
	arrayName.sort()

	dic = {}

	for m in arrayName:
		count+=1
		dic[count] = m

	min_number = arrayNumber[0]
	max_number = arrayNumber[-1]

	ram = random.randint(min_number, max_number)

	# print(ram)
	# print(arrayName)
	print(dic)

	nameSound = dic[ram]

	name = "./sounds/"+nameSound

	mixer.init()
	mixer.music.load(name)
	mixer.music.play()