import numpy as np

list = [[[ 1.01957897e+00, -1.43690480e-02,  3.18369123e+02],
 [ 9.22538236e-03,  9.16220425e-01,  1.73148594e+02],
 [-1.30045430e-06, -2.07484577e-05,  1.00000000e+00]],\
[[ 1.07982464e+00,  4.62337069e-03,  4.12049433e+02],
 [ 3.18922820e-02,  9.99820123e-01,  1.79961225e+02],
 [ 7.81124980e-06, -3.11166590e-07,  1.00000000e+00]],\
[[ 1.04595712e+00,  1.03551126e-02,  2.93347704e+01],
 [ 1.43899550e-02,  1.02704200e+00,  1.11245063e+02],
 [ 1.35892934e-05,  6.52359817e-06,  1.00000000e+00]],\
[[ 1.00501163e+00,  4.01320054e-02, -9.72043756e+01],
 [-2.60083916e-02,  1.00281698e+00, -4.33996708e+01],
 [-4.10961767e-07,  5.29408472e-06,  1.00000000e+00]],\
[[ 1.06010206e+00,  7.94848544e-02, -9.64612292e+00],
 [-3.21908936e-02,  1.09772268e+00, -8.57154756e+01],
 [-2.63796756e-05,  6.19480028e-05,  1.00000000e+00]],\
[[ 1.01121814e+00,  8.90149321e-03,  4.24079756e+02],
 [-3.35466173e-02,  9.53905165e-01,  3.40407006e+01],
 [-8.68987294e-06, -1.17645061e-05,  1.00000000e+00]],\
[[ 1.05296885e+00, -3.26900266e-02,  6.15620508e+01],
 [ 2.25776055e-02,  1.00638138e+00,  1.11265862e+01],
 [ 9.32012394e-06, -7.90713042e-06,  1.00000000e+00]],\
[[ 1.00000000e+00,  1.98406941e-13, -1.16177714e-11],
 [ 5.67253073e-17,  1.00000000e+00, -2.39985743e-13],
 [ 1.20086894e-18,  1.15715100e-16,  1.00000000e+00]],\
[[ 1.16290332e+00,  1.69366669e-02,  3.86240531e+02],
 [ 7.47575258e-03,  1.03127674e+00, -1.41563029e+02],
 [ 1.25885490e-05,  2.58888570e-06,  1.00000000e+00]],\
[[ 1.06825386e+00, -3.23315712e-03,  3.44306233e+02],
 [-5.31095155e-02,  9.54797203e-01, -1.64883348e+01],
 [ 2.24522711e-06, -2.15206102e-05,  1.00000000e+00]],\
[[ 1.00438170e+00, -3.00770953e-02,  3.88665598e+02],
 [-2.64264338e-02,  9.62697039e-01, -4.27716065e+01],
 [-7.99140159e-06, -1.05289934e-05,  1.00000000e+00]],\
[[ 1.04601971e+00,  6.39234610e-03, -6.11293428e+01],
 [-2.84425510e-02,  1.01440933e+00, -6.44636876e+01],
 [ 8.68409686e-06, -4.18009900e-06,  1.00000000e+00]],\
[[ 9.27693644e-01,  1.52155540e-02,  3.69926810e+02],
 [-3.54698583e-02,  9.62638888e-01,  2.70937029e+02],
 [-1.82760085e-05, -1.75273170e-06,  1.00000000e+00]],\
[[ 8.39206177e-01,  1.88206198e-02,  4.06179276e+02],
 [-2.52947059e-02,  9.47559652e-01,  2.82484085e+02],
 [-2.92776984e-05,  3.16080753e-06,  1.00000000e+00]],\
[[ 9.71058651e-01, -4.02237210e-03,  1.07615563e+02],
 [-4.05066091e-02,  9.93717235e-01,  2.36036650e+00],
 [-3.27255701e-06, -3.87909943e-06,  1.00000000e+00]],\
[[ 9.03296270e-01,  6.08858410e-03,  8.28859836e+01],
 [-5.59638117e-02,  9.68974099e-01, -7.87302133e+00],
 [-1.36556976e-05, -2.21980704e-06,  1.00000000e+00]]]

a = np.array(list)
np.save('H_result_add.npy', a)