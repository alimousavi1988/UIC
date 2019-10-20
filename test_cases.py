""" This file contains the test cases. You can replace each test case in
the data() function. """

######### Linear ##########

num_samples = 200
x = np.random.rand(num_samples, 1)
y = x

######### Parabolic ##########

num_samples = 200
x = np.random.rand(num_samples, 1)
y = np.power(x, 2)

######### Cubic ##########

num_samples = 200
x = np.random.rand(num_samples, 1)
y = np.power(x, 3)

######### Periodic ##########

num_samples = 200
x1 = np.random.rand(int(num_samples/3), 1) * 0.33
x2 = np.random.rand(int(num_samples/3), 1) * 0.33 + 0.33
x3 = np.random.rand(int(num_samples/3), 1) * 0.33 + 0.66
y1 = x1
y2 = x2 - 0.33
y3 = x3 - 0.66
x=np.concatenate((x1, x2, x3)).reshape(num_samples,1)
y=np.concatenate((y1, y2, y3)).reshape(num_samples,1)

######### Sin (single freq) ##########

num_samples = 200
x = np.random.rand(num_samples, 1)
y = np.sin(4 * np.pi * x)

######### Sin (diff. freq) ##########

num_samples = 200
x1 = np.random.rand(int(num_samples/2), 1) * 0.5
y1 = np.sin(4 * np.pi * x1)
x2 = np.random.rand(int(num_samples/2), 1) * 0.5 + 0.5
y2 = np.sin(8 * np.pi * x2 + np.pi/2)
x=np.concatenate((x1,x2)).reshape(num_samples,1)
y=np.concatenate((y1,y2)).reshape(num_samples,1)

######### Chirp ##########

num_samples = 200
x = np.random.rand(num_samples, 1) * 4 * np.pi
y = np.sin(np.power(x, 1.6))

######### Random ##########

num_samples = 200
x = np.random.rand(num_samples, 1)
y = np.random.rand(num_samples, 1)

######### Two Lines ##########

num_samples = 200
x1 = np.random.rand(int(num_samples/2), 1)
y1 = x1
x2 = np.random.rand(int(num_samples/2), 1)
y2 = 3 * x2
x=np.concatenate((x1,x2)).reshape(num_samples,1)
y=np.concatenate((y1,y2)).reshape(num_samples,1)

######### Sinusoidal Mixture ##########

num_samples = 200
x1 = np.random.rand(int(num_samples/2), 1)
y1 = np.sin(4*np.pi*x1)
x2 = np.random.rand(int(num_samples/2), 1)
y2 = np.sin(8*np.pi*x2)
x=np.concatenate((x1,x2)).reshape(num_samples,1)
y=np.concatenate((y1,y2)).reshape(num_samples,1)

######### Circle ##########

num_samples = 200
x1 = np.random.rand(int(num_samples/2), 1)
y1 = np.sqrt(0.25-np.power((x1-0.5),2))+0.5
x2 = np.random.rand(int(num_samples/2), 1)
y2 = -np.sqrt(0.25-np.power((x2-0.5),2))+0.5
x=np.concatenate((x1,x2)).reshape(num_samples,1)
y=np.concatenate((y1,y2)).reshape(num_samples,1)

######### Noisy Circle ##########

num_samples = 200
x1 = np.random.rand(int(num_samples/2), 1)
y1 = np.sqrt(0.25 - np.power((x1 - 0.5), 2))+0.5
x2 = np.random.rand(int(num_samples/2), 1)
y2 = -np.sqrt(0.25 - np.power((x2 - 0.5), 2))+0.5
x = np.concatenate((x1, x2)).reshape(num_samples,1)
y = np.concatenate((y1, y2)).reshape(num_samples,1)
y = y + np.random.rand(num_samples, 1) * 0.1 - 0.05


######### Multidimensional Linear ##########

num_samples = 5000
x = np.random.rand(num_samples, 2)
y = (x[:,0] + x[:,1]).reshape(num_samples,1)

######### Multidimensional Parabolic ##########

num_samples = 5000

x = np.random.rand(num_samples, 2)
y = (np.power(x[:,0],2) + 100*np.power(x[:,1],2)).reshape(num_samples,1)


######### Multidimensional Cubic ##########

num_samples = 5000
x = np.random.rand(num_samples, 2)
y = (np.power((x[:,0] - 0.5),3) + np.power((x[:,1] - 0.5),3)).reshape(num_samples,1)


######### Multidimensional Sinusoidal ##########

num_samples = 5000
x = np.random.rand(num_samples, 2) * 2 * np.pi
y = np.sin(x[:,0] + x[:,1]).reshape(num_samples,1)


######### Multidimensional Multi-Freq Sinusoidal  ##########

num_samples = 5000

x1 = np.random.rand(int(num_samples/2), 2) * 2 * np.pi
x2 = np.random.rand(int(num_samples/2), 2) * 2 * np.pi + 2 * np.pi
y1 = np.sin(x1[:,0]+x1[:,1])
y2 = np.sin(1.5 * (x2[:,0] + x2[:,1]))
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2)).reshape(num_samples,1)

######### Multidimensional Independent ##########

num_samples = 5000
x = np.random.rand(num_samples, 2)
y = np.random.rand(num_samples, 1).reshape(num_samples,1)

######### Multidimensional Periodic ##########

num_samples = 5000
x1 = np.random.rand(int(num_samples/3), 2) * 5
x2 = np.random.rand(int(num_samples/3), 2) * 5 + 5
x3 = np.random.rand(int(num_samples/3), 2) * 5 + 10
y1 = x1[:,0] + x1[:,1]
y2 = x2[:,0] + x2[:,1] - 10
y3 = x3[:,0] + x3[:,1] - 20
x = np.concatenate((x1, x2, x3))
y = np.concatenate((y1, y2, y3)).reshape(num_samples,1)

######### Multidimensional Non-functional Plates ##########

num_samples = 5000
x1 = np.random.rand(int(num_samples/2), 2)
x2 = np.random.rand(int(num_samples/2), 2)
y1 = 4 * (x1[:,0] + x1[:,1])
y2 = -4 * (x2[:,0] + x2[:,1]) + 8
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2)).reshape(num_samples,1)


######### Multidimensional Non-Functional Sinusoidal ##########

num_samples = 5000
x1 = np.random.rand(int(num_samples/2), 2) * 2 * np.pi
x2 = np.random.rand(int(num_samples/2), 2) * 2 * np.pi
y1 = np.sin(x1[:,0]+x1[:,1])
y2 = np.sin(4*(x2[:,0]+x2[:,1]))
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2)).reshape(num_samples,1)

######### Mixed Multidimensional ##########

# For this multidimensionsl problem set the 'grid_factor' to 0.8

num_samples = 10**5
x = np.random.rand(num_samples, 5)
y1 = x[:, 0].reshape(num_samples, 1)
y2 = np.power(x[:, 1], 2).reshape(num_samples, 1)
y3 = np.power(x[:, 2], 3).reshape(num_samples, 1)
y4 = np.sin(x[:, 3]).reshape(num_samples, 1)
y5 = np.cos(x[:, 4]).reshape(num_samples, 1)
y = np.concatenate((y1, y2, y3, y4, y5),axis=1)

