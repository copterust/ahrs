# State vector
* q0..q3: quaternion
* bx, by, bz

# System dynamics
q_dot = 1/2 K(w)q

To account for gyro bias we substract b:

q_dot = 1/2 K(w-b)q

Linearization

q_dot = (q(k+1)-q(k))/T

Next state:
```
| q |     | I_4x4  -T/2K(q) | | q |    | T/2K(q) |
|   |  =  |                 | |   |  + |         |  *  w_k
| b |k+1  | 0_3x4   I_3x3   | | b |k   | 0_3x3   |k
```

# Accelerometer
We use gravity vector as reference. As we know body rotation we could
predict the acceleration that will be measured. An actual measurement
could be used to get an error in our orientation estimate.

a = R(-g) * e_a + b
a = -gh(q) + c

As it is non-linear functions linearize it using Jacobian (gradient) with Taylor expansion.

# Magnetometer
We rotate 3d magnetometer and remove Z as we only interested in direction.
